import { EXACT_UUID_RE, splitTextOnUuids, UUID_REGEX } from "~/utils/uuid";

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

interface MdastLink {
  type: "link";
  url: string;
  children: MdastNode[];
}

interface MdastParent {
  type: string;
  children: MdastNode[];
}

type MdastNode =
  | MdastText
  | MdastInlineCode
  | MdastLink
  | MdastParent
  | { type: string };

function uuidLink(uuid: string): MdastLink {
  return {
    type: "link",
    url: `#uuid:${uuid}`,
    children: [{ type: "text", value: uuid }],
  };
}

/**
 * Remark plugin that transforms UUID patterns into link nodes
 * with `#uuid:` fragment URLs (fragment identifiers survive
 * ReactMarkdown's URL sanitization, unlike custom protocols).
 *
 * Handles:
 * - UUIDs in plain text nodes (split into text + link fragments)
 * - `inlineCode` nodes whose entire value is a UUID (replaced with a link)
 *
 * Correctly skips:
 * - Link URLs (the `url` property, not in children)
 * - Fenced code blocks (`code` nodes)
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
        newChildren.push(uuidLink(codeNode.value.trim()));
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
 * Split a text MDAST node into text + link nodes using the shared
 * `splitTextOnUuids` utility. Returns the original node unchanged
 * if no UUIDs are found.
 */
function splitTextToMdast(textNode: MdastText): MdastNode[] {
  // Quick check: does the text contain anything UUID-like?
  UUID_REGEX.lastIndex = 0;
  if (!UUID_REGEX.test(textNode.value)) {
    return [textNode];
  }

  const segments = splitTextOnUuids(textNode.value);
  return segments.map(
    (seg): MdastNode =>
      seg.isUuid ? uuidLink(seg.text) : { type: "text", value: seg.text },
  );
}
