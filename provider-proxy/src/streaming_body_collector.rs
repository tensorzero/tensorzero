use pin_project::pinned_drop;

pub type DoneCallback = Box<dyn FnOnce(BytesMut) + Send + Sync>;

/// A helper type that runs a callback once the whole body has been received.
/// Unlike `Collected`, this clones the frames we receive from the underlying body,
/// while still forwarding them to the caller. This allows it to be used with streaming bodies
/// without causing the entire stream to block.
#[pin_project::pin_project(PinnedDrop)]
pub struct StreamingBodyCollector<T> {
    #[pin]
    body: T,
    buffer: BytesMut,
    done_callback: Option<DoneCallback>,
}

use std::{
    pin::Pin,
    task::{Context, Poll, ready},
};

use bytes::{Bytes, BytesMut};
use hyper::body::{Body, Frame, SizeHint};

impl<T> StreamingBodyCollector<T> {
    pub fn new(body: T, cb: DoneCallback) -> Self {
        Self {
            body,
            buffer: BytesMut::new(),
            done_callback: Some(cb),
        }
    }
}

impl<T: Body<Data = Bytes>> Body for StreamingBodyCollector<T>
where
    T::Error: std::fmt::Debug,
{
    type Data = T::Data;
    type Error = T::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.as_mut().project();
        let frame = this.body.poll_frame(cx);
        let frame = ready!(frame);
        match &frame {
            Some(Ok(frame)) => {
                assert!(
                    !this.done_callback.is_none(),
                    "poll_frame got frame after done_callback was called"
                );

                if let Some(data) = frame.data_ref() {
                    this.buffer.extend(data);
                } else {
                    // If we ever get this, we'll need to decide how to handle caching it.
                    panic!("Unexpected frame: {frame:?}");
                }
                if self.body.is_end_stream() {
                    let this = self.project();
                    if let Some(cb) = this.done_callback.take() {
                        cb(this.buffer.clone());
                    }
                }
            }
            None => {
                if let Some(cb) = this.done_callback.take() {
                    cb(self.buffer.clone());
                }
            }
            _ => {}
        }
        Poll::Ready(frame)
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

#[pinned_drop]
impl<T> PinnedDrop for StreamingBodyCollector<T> {
    fn drop(self: Pin<&mut Self>) {
        let this = self.project();
        if let Some(cb) = this.done_callback.take() {
            cb(this.buffer.clone());
        }
    }
}
