import time
from playwright.sync_api import sync_playwright

def test_history_and_details():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        print("1. Navigating to Safe-Bit dashboard (http://localhost:3000)...")
        page.goto("http://localhost:3000", wait_until="networkidle")
        time.sleep(2)

        # 2. Check that Header has Live Stream and Transaction History buttons
        print("2. Verifying navigation buttons...")
        btn_live = page.locator("#nav-live-stream")
        btn_hist = page.locator("#nav-history-log")
        assert btn_live.is_visible(), "Live Stream button not found!"
        assert btn_hist.is_visible(), "Transaction History button not found!"

        # 3. Switch to Transaction History view
        print("3. Switching to Transaction History view...")
        btn_hist.click()
        time.sleep(2)

        # Verify History Explorer is active
        header_title = page.locator("span:has-text('Transaction History Explorer')")
        assert header_title.is_visible(), "Transaction History Explorer title not visible!"

        # Verify table has rows
        rows = page.locator("table tbody tr")
        row_count = rows.count()
        print(f"Loaded {row_count} transaction rows in History Explorer.")
        assert row_count > 0, "No transaction rows loaded in History Explorer!"

        # Take screenshot of History Explorer (Dark Theme)
        hist_dark_path = "C:/Users/bhuva/.gemini/antigravity/brain/27eb4491-392f-4ac1-b938-042b73917f55/safebit_history_page_dark.png"
        page.screenshot(path=hist_dark_path)
        print(f"History Dark screenshot saved to {hist_dark_path}")

        # 4. Test Search Filtering
        print("4. Testing Search Filtering...")
        search_input = page.locator("input[placeholder*='Search by Tx ID']")
        first_tx_id = page.locator("table tbody tr:first-child td:first-child div:first-child span").inner_text()
        print(f"First transaction ID: {first_tx_id}")
        search_input.fill(first_tx_id)
        time.sleep(1)
        filtered_count = page.locator("table tbody tr").count()
        print(f"Filtered rows for '{first_tx_id}': {filtered_count}")
        assert filtered_count >= 1

        # Clear search
        search_input.fill("")
        time.sleep(1)

        # 5. Test Clicking on a Transaction to open Detailed Forensic Dossier
        print("5. Clicking on a transaction to inspect ALL details...")
        first_row = page.locator("table tbody tr").first
        first_row.click()
        time.sleep(1.5)

        # Verify Detail Modal opened
        dossier_title = page.locator("span:has-text('Transaction Dossier')")
        assert dossier_title.is_visible(), "Transaction Dossier Modal not opened!"

        # Check Hero KPI Cards (Mule Risk, Amount, SLA Latency, Louvain Ring ID)
        assert page.locator("span:has-text('Calibrated Mule Risk')").is_visible()
        assert page.locator("span:has-text('Transaction Volume')").is_visible()
        assert page.locator("span:has-text('SLA Latency')").is_visible()
        assert page.locator("span:has-text('Louvain Ring ID')").is_visible()

        # Check Tabs in Modal
        tabs = [
          ("Counterparty & Telemetry", "Originating Sender"),
          ("2-Hop Ego-Subgraph", "2-Hop Bounded Ego-Subgraph"),
          ("Neural & Invariants", "Neural Model Backbone & Inferences"),
          ("Audit JSON", "Raw FinCEN / Safe-Bit Verification Payload")
        ]

        for tab_name, expected_content in tabs:
            print(f"Testing modal tab: {tab_name}...")
            tab_btn = page.locator(f"button:has-text('{tab_name}')")
            assert tab_btn.is_visible(), f"Tab button {tab_name} not found!"
            tab_btn.click()
            time.sleep(0.8)
            content_elem = page.locator(f"*:has-text('{expected_content}')").first
            assert content_elem.is_visible(), f"Content '{expected_content}' not found in tab {tab_name}!"

        # Take screenshot of Transaction Detail Modal
        detail_screenshot_path = "C:/Users/bhuva/.gemini/antigravity/brain/27eb4491-392f-4ac1-b938-042b73917f55/safebit_transaction_detail_modal.png"
        page.screenshot(path=detail_screenshot_path)
        print(f"Transaction detail modal screenshot saved to {detail_screenshot_path}")

        # Close Modal
        print("Closing Transaction Detail Modal...")
        page.locator("div.fixed button:has(svg.lucide-x)").first.click()
        time.sleep(0.5)
        assert not page.locator("span:has-text('Transaction Dossier')").is_visible()

        # 6. Test Light Mode Toggle on History Page
        print("6. Testing Light Mode on History Page...")
        theme_toggle = page.locator("#theme-toggle")
        theme_toggle.click()
        time.sleep(1.5)

        # Verify Light Mode applied
        is_light = page.evaluate("() => document.documentElement.classList.contains('light')")
        assert is_light is True, "Light theme class not present!"

        hist_light_path = "C:/Users/bhuva/.gemini/antigravity/brain/27eb4491-392f-4ac1-b938-042b73917f55/safebit_history_page_light.png"
        page.screenshot(path=hist_light_path)
        print(f"History Light screenshot saved to {hist_light_path}")

        # Open Detail Modal in Light Mode
        print("Opening Detail Modal in Light Mode...")
        page.locator("table tbody tr").first.click()
        time.sleep(1.5)
        assert page.locator("span:has-text('Transaction Dossier')").is_visible()

        detail_light_path = "C:/Users/bhuva/.gemini/antigravity/brain/27eb4491-392f-4ac1-b938-042b73917f55/safebit_transaction_detail_light.png"
        page.screenshot(path=detail_light_path)
        print(f"Detail modal Light screenshot saved to {detail_light_path}")

        # Close Modal in Light Mode
        page.locator("div.fixed button:has(svg.lucide-x)").first.click()
        time.sleep(0.5)

        # 7. Return to Live Stream View
        print("7. Returning to Live Stream View...")
        page.locator("button:has-text('Return to Live Stream')").click()
        time.sleep(1.5)
        assert page.locator("h2:has-text('Recent transactions')").is_visible()
        print("Live Stream view confirmed.")

        # Toggle back to Dark Mode for persistence
        theme_toggle.click()
        time.sleep(1)

        print(f"Total console errors: {len(console_errors)}")
        if console_errors:
            print("Errors:", console_errors[:5])

        browser.close()
        print("ALL HISTORY PAGE AND TRANSACTION DETAIL TESTS PASSED!")

if __name__ == "__main__":
    test_history_and_details()

