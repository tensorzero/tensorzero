export abstract class ErrorWithStatus extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export class BadRequestError extends ErrorWithStatus {
  constructor(message: string) {
    super(message, 400);
    this.name = "BadRequestError";
  }
}

export class NotFoundError extends ErrorWithStatus {
  constructor(message: string) {
    super(message, 404);
    this.name = "NotFoundError";
  }
}

export class InternalServerError extends ErrorWithStatus {
  constructor(message: string) {
    super(message, 500);
    this.name = "InternalServerError";
  }
}
