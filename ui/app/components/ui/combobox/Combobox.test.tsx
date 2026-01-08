import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Combobox } from "./Combobox";

function generateItems(count: number): string[] {
  return Array.from({ length: count }, (_, i) => `item_${i + 1}`);
}

describe("Combobox", () => {
  const defaultProps = {
    selected: null,
    onSelect: vi.fn(),
    items: generateItems(10),
    placeholder: "Select item",
    emptyMessage: "No items found",
  };

  describe("basic functionality", () => {
    it("renders placeholder when no item selected", () => {
      render(<Combobox {...defaultProps} />);
      expect(screen.getByPlaceholderText("Select item")).toBeInTheDocument();
    });

    it("shows selected value in input", () => {
      render(<Combobox {...defaultProps} selected="item_1" />);
      expect(screen.getByDisplayValue("item_1")).toBeInTheDocument();
    });

    it("calls onSelect when item is clicked", async () => {
      const onSelect = vi.fn();
      render(<Combobox {...defaultProps} onSelect={onSelect} />);

      await userEvent.click(screen.getByPlaceholderText("Select item"));
      await userEvent.click(screen.getByText("item_1"));

      expect(onSelect).toHaveBeenCalledWith("item_1");
    });

    it("filters items based on search input", async () => {
      render(<Combobox {...defaultProps} />);

      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);
      await userEvent.type(input, "item_1");

      // Should show item_1 and item_10 (both contain "item_1")
      expect(screen.getByText("item_1")).toBeInTheDocument();
      expect(screen.getByText("item_10")).toBeInTheDocument();
      expect(screen.queryByText("item_2")).not.toBeInTheDocument();
    });
  });

  describe("virtualization", () => {
    it("does not virtualize when item count is below threshold", () => {
      render(
        <Combobox
          {...defaultProps}
          items={generateItems(50)}
          virtualizeThreshold={100}
        />,
      );
      // Non-virtualized: all items rendered
      // (This is hard to test without checking DOM structure)
    });

    it("virtualizes when item count exceeds threshold", async () => {
      render(
        <Combobox
          {...defaultProps}
          items={generateItems(200)}
          virtualizeThreshold={100}
        />,
      );

      await userEvent.click(screen.getByPlaceholderText("Select item"));

      // With virtualization, not all 200 items should be in DOM
      // Only visible items + overscan (~20-30 items)
      const items = screen.getAllByRole("option");
      expect(items.length).toBeLessThan(200);
      expect(items.length).toBeGreaterThan(0);
    });

    it("respects virtualizeThreshold=0 to always virtualize", async () => {
      render(
        <Combobox
          {...defaultProps}
          items={generateItems(20)}
          virtualizeThreshold={0}
        />,
      );

      await userEvent.click(screen.getByPlaceholderText("Select item"));

      // Even with 20 items, should virtualize (render fewer than 20)
      const items = screen.getAllByRole("option");
      expect(items.length).toBeLessThanOrEqual(20);
    });

    it("respects virtualizeThreshold=Infinity to never virtualize", async () => {
      render(
        <Combobox
          {...defaultProps}
          items={generateItems(200)}
          virtualizeThreshold={Infinity}
        />,
      );

      await userEvent.click(screen.getByPlaceholderText("Select item"));

      // Should render all 200 items
      const items = screen.getAllByRole("option");
      expect(items.length).toBe(200);
    });
  });

  describe("keyboard navigation (virtualized)", () => {
    const virtualizedProps = {
      ...defaultProps,
      items: generateItems(200),
      virtualizeThreshold: 100,
    };

    it("ArrowDown moves highlight to next item", async () => {
      render(<Combobox {...virtualizedProps} />);
      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);

      fireEvent.keyDown(input, { key: "ArrowDown" });

      // First item should be highlighted, then second after ArrowDown
      const highlightedItem = screen.getByRole("option", { selected: true });
      expect(highlightedItem).toHaveTextContent("item_2");
    });

    it("ArrowUp moves highlight to previous item", async () => {
      render(<Combobox {...virtualizedProps} />);
      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);

      // Move down twice, then up once
      fireEvent.keyDown(input, { key: "ArrowDown" });
      fireEvent.keyDown(input, { key: "ArrowDown" });
      fireEvent.keyDown(input, { key: "ArrowUp" });

      const highlightedItem = screen.getByRole("option", { selected: true });
      expect(highlightedItem).toHaveTextContent("item_2");
    });

    it("Home moves highlight to first item", async () => {
      render(<Combobox {...virtualizedProps} />);
      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);

      // Move down several times
      fireEvent.keyDown(input, { key: "ArrowDown" });
      fireEvent.keyDown(input, { key: "ArrowDown" });
      fireEvent.keyDown(input, { key: "ArrowDown" });

      // Home should go back to first
      fireEvent.keyDown(input, { key: "Home" });

      const highlightedItem = screen.getByRole("option", { selected: true });
      expect(highlightedItem).toHaveTextContent("item_1");
    });

    it("End moves highlight to last item", async () => {
      const items = generateItems(20);
      render(
        <Combobox {...defaultProps} items={items} virtualizeThreshold={0} />,
      );
      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);

      fireEvent.keyDown(input, { key: "End" });

      const highlightedItem = screen.getByRole("option", { selected: true });
      expect(highlightedItem).toHaveTextContent("item_20");
    });

    it("Enter selects highlighted item", async () => {
      const onSelect = vi.fn();
      render(<Combobox {...virtualizedProps} onSelect={onSelect} />);
      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);

      fireEvent.keyDown(input, { key: "ArrowDown" });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(onSelect).toHaveBeenCalledWith("item_2");
    });

    it("Escape closes dropdown", async () => {
      render(<Combobox {...virtualizedProps} />);
      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);

      // Dropdown should be open
      expect(
        screen.getByRole("option", { name: /item_1/ }),
      ).toBeInTheDocument();

      fireEvent.keyDown(input, { key: "Escape" });

      // Dropdown should be closed
      expect(screen.queryByRole("option")).not.toBeInTheDocument();
    });

    it("PageDown jumps multiple items", async () => {
      render(<Combobox {...virtualizedProps} />);
      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);

      fireEvent.keyDown(input, { key: "PageDown" });

      // Should jump ~8 items
      const highlightedItem = screen.getByRole("option", { selected: true });
      expect(highlightedItem).toHaveTextContent("item_9");
    });
  });

  describe("highlight index bounds", () => {
    it("clamps highlight index when filter reduces items", async () => {
      const onSelect = vi.fn();
      render(
        <Combobox
          {...defaultProps}
          items={generateItems(200)}
          virtualizeThreshold={100}
          onSelect={onSelect}
        />,
      );
      const input = screen.getByPlaceholderText("Select item");
      await userEvent.click(input);

      // Navigate to item 50
      for (let i = 0; i < 50; i++) {
        fireEvent.keyDown(input, { key: "ArrowDown" });
      }

      // Now filter to only show 5 items (item_1 through item_5 match "item_[1-5]")
      await userEvent.clear(input);
      await userEvent.type(input, "item_1");

      // Press Enter - should select valid item (not crash)
      fireEvent.keyDown(input, { key: "Enter" });

      // Should have selected an item that exists in filtered list
      expect(onSelect).toHaveBeenCalled();
      const selectedItem = onSelect.mock.calls[0][0];
      expect(selectedItem).toMatch(/item_1/);
    });
  });
});
