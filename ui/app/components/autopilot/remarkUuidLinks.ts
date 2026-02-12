import { EXACT_UUID_RE, splitTextOnUuids } from "~/utils/uuid";

/**
 * The custom HTML element name used to bridge remark (AST) to React.
 * Remark plugin sets `data.hName` to this value; the component map
 * in the consumer maps this element name to the `UuidLink` component.
 */
export const UUID_LINK_ELEMENT = "uuid-link";

/**
 * MDAST node types used by this plugin.
 * Minimal definitions to avoid a dependency on @types/mdast.
 */
interface MdastText {
  type: "text";
  value: string;
}

interface MdastInlineCode {
  type: "inlineCode";
  value: string;
}

interface MdastParent {
  type: string;
  children: MdastNode[];
}

type MdastNode =
  | MdastText
  | MdastInlineCode
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

/**
 * Remark plugin that transforms UUID patterns into custom `uuidLink` nodes.
 * Uses `data.hName` / `data.hProperties` to bridge to React — the consumer
 * maps the `UUID_LINK_ELEMENT` element name to a React component.
 *
 * Handles:
 * - UUIDs in plain text nodes (split into text + uuidLink fragments)
 * - `inlineCode` nodes whose entire value is a UUID (replaced with uuidLink)
 *
 * Skips:
 * - Fenced code blocks (`code` nodes)
 * - Existing links (`link` nodes)
 * - Inline code containing mixed content (only pure-UUID inline code is linked)
 */
export function remarkUuidLinks() {
  return (tree: MdastParent) => {
    visitParent(tree);
  };
}

/** Node types whose content should NOT be transformed. */
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
      const codeNode = child as MdastInlineCode;
      if (EXACT_UUID_RE.test(codeNode.value)) {
        newChildren.push(uuidLinkNode(codeNode.value.trim()));
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

/**
 * Split a text MDAST node into text + uuidLink nodes.
 * Returns the original node unchanged if no UUIDs are found.
 */
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
