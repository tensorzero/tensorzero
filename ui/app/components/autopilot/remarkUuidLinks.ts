import { splitTextOnUuids } from "~/utils/uuid";

/** Custom element name bridging remark AST to React via `data.hName`. */
export const UUID_LINK_ELEMENT = "uuid-link";

/** Minimal MDAST types to avoid a `@types/mdast` dependency. */
interface MdastText {
  type: "text";
  value: string;
}

interface MdastParent {
  type: string;
  children: MdastNode[];
}

type MdastNode =
  | MdastText
  | MdastParent
  | { type: string; data?: Record<string, unknown> };

function uuidLinkNode(uuid: string): MdastNode {
  return {
    type: "uuidLink",
    data: {
      hName: UUID_LINK_ELEMENT,
      hProperties: { uuid },
    },
    children: [{ type: "text", value: uuid }],
  } as MdastNode;
}

/** Remark plugin that transforms UUID patterns into custom `uuidLink` nodes. */
export function remarkUuidLinks() {
  return (tree: MdastParent) => {
    visitParent(tree);
  };
}

const SKIP_TYPES = new Set(["code", "link"]);

function visitParent(node: MdastParent) {
  const newChildren: MdastNode[] = [];
  let changed = false;

  for (const child of node.children) {
    if (child.type === "text") {
      const parts = splitTextToMdast(child as MdastText);
      if (parts.length === 1 && parts[0] === child) {
        newChildren.push(child);
      } else {
        newChildren.push(...parts);
        changed = true;
      }
    } else if (child.type === "inlineCode") {
      const segments = splitTextOnUuids((child as MdastText).value);
      if (segments.length === 1 && segments[0].isUuid) {
        newChildren.push(uuidLinkNode(segments[0].text));
        changed = true;
      } else {
        newChildren.push(child);
      }
    } else if (SKIP_TYPES.has(child.type)) {
      newChildren.push(child);
    } else {
      if ("children" in child && Array.isArray(child.children)) {
        visitParent(child as MdastParent);
      }
      newChildren.push(child);
    }
  }

  if (changed) {
    node.children = newChildren;
  }
}

function splitTextToMdast(textNode: MdastText): MdastNode[] {
  const segments = splitTextOnUuids(textNode.value);
  if (!segments.some((s) => s.isUuid)) {
    return [textNode];
  }
  return segments.map(
    (seg): MdastNode =>
      seg.isUuid ? uuidLinkNode(seg.text) : { type: "text", value: seg.text },
  );
}
