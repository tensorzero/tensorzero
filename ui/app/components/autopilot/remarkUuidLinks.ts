import { UUID_REGEX } from "~/components/autopilot/RichText";

/**
 * MDAST node types used by this plugin.
 * Minimal definitions to avoid a dependency on @types/mdast.
 */
interface MdastText {
  type: "text";
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

type MdastNode = MdastText | MdastLink | MdastParent | { type: string };

/**
 * Remark plugin that transforms UUID patterns in text nodes into
 * link nodes with `uuid://` scheme URLs.
 *
 * Operates at the AST level, so it correctly skips:
 * - Link URLs (the `url` property, not in children)
 * - Fenced code blocks (`code` nodes)
 * - Inline code (`inlineCode` nodes)
 */
export function remarkUuidLinks() {
  return (tree: MdastParent) => {
    visitParent(tree);
  };
}

/** Node types whose content should NOT be transformed. */
const SKIP_TYPES = new Set(["code", "inlineCode", "link"]);

function visitParent(node: MdastParent) {
  const newChildren: MdastNode[] = [];
  let changed = false;

  for (const child of node.children) {
    if (child.type === "text") {
      const parts = splitTextOnUuids(child as MdastText);
      if (parts.length === 1 && parts[0] === child) {
        newChildren.push(child);
      } else {
        newChildren.push(...parts);
        changed = true;
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

function splitTextOnUuids(textNode: MdastText): MdastNode[] {
  const text = textNode.value;
  UUID_REGEX.lastIndex = 0;

  const results: MdastNode[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(UUID_REGEX)) {
    const matchStart = match.index;
    const matchEnd = matchStart + match[0].length;

    if (matchStart > lastIndex) {
      results.push({ type: "text", value: text.slice(lastIndex, matchStart) });
    }

    results.push({
      type: "link",
      url: `uuid://${match[0]}`,
      children: [{ type: "text", value: match[0] }],
    });

    lastIndex = matchEnd;
  }

  if (lastIndex < text.length) {
    results.push({ type: "text", value: text.slice(lastIndex) });
  }

  return results.length > 0 ? results : [textNode];
}
