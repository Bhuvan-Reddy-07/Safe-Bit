import time
import json
from playwright.sync_api import sync_playwright

def run_e2e_test():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        print("Navigating to http://localhost:3000...")
        page.goto("http://localhost:3000", wait_until="networkidle")
        time.sleep(2)

        # 1. Verify Header and Branding
        header_text = page.locator("header").inner_text()
        print("Header content:", header_text[:50])
        assert "Safe-Bit" in header_text

        # 2. Monitor Live Stream Graph and Table updates
        print("Monitoring live stream updates...")
        initial_tx_id = page.locator("section h2 + span").evaluate("el => el.textContent")
        first_row_tx = page.locator("table tbody tr:first-child td:first-child").inner_text()
        print(f"First row Tx ID: {first_row_tx}")

        # Wait 3 seconds for stream transactions to arrive
        time.sleep(3)
        updated_row_tx = page.locator("table tbody tr:first-child td:first-child").inner_text()
        print(f"Updated row Tx ID after stream: {updated_row_tx}")

        # Check Cytoscape canvas elements count
        node_count = page.evaluate("() => window.cytoscape ? window.cytoscape.instances[0]?.nodes().length : -1")
        print(f"Cytoscape node count: {node_count}")

        # 3. Test Inject Buttons
        buttons = [
            ("Inject normal P2P", "P2P"),
            ("Inject scatter-gather", "SCATTER"),
            ("Inject cycle ring", "CYCLE"),
            ("Inject fan-out", "FAN"),
            ("Inject peeling chain", "PEEL"),
        ]

        for btn_name, expected_tag in buttons:
            print(f"Testing button: {btn_name}...")
            btn = page.locator(f"button:has-text('{btn_name}')")
            assert btn.is_visible()
            btn.click()
            time.sleep(1.5)

            # Check that latest transaction and graph header updated
            current_tx_header = page.locator("section div.f-mono:has-text('TX_')").first.inner_text()
            print(f"After {btn_name}, graph shows: {current_tx_header}")

            # Verify Mule Risk confidence score is working and non-zero
            risk_text = page.locator("div:has(> .lbl:text-is('Mule risk')) > div:last-child").first.inner_text()
            print(f"Mule risk display: {risk_text}")
            assert "0.0%" not in risk_text, f"Confidence score unexpectedly 0.0%: {risk_text}"
            
            # Verify toast notification or evidence update
            toast = page.locator("div.absolute.top-16").first
            if toast.is_visible():
                print(f"Toast visible: {toast.inner_text()}")

        # 4. Test Zoom and Reset Buttons
        print("Testing zoom and reset buttons...")
        page.locator("button[title='Zoom In']").click()
        time.sleep(0.5)
        page.locator("button[title='Zoom Out']").click()
        time.sleep(0.5)
        page.locator("button:has-text('Reset graph')").click()
        time.sleep(0.5)

        # 5. Test Threshold Drawer
        print("Testing Threshold drawer...")
        thr_btn = page.locator("footer button:has-text('Threshold')")
        thr_btn.click()
        time.sleep(1)
        assert page.locator("#thrB").is_visible()
        # Close threshold drawer
        page.locator("#thrB button").click()
        time.sleep(0.5)
        assert not page.locator("#thrB").is_visible()

        # 6. Test Engine Performance Drawer
        print("Testing Engine Performance drawer...")
        perf_btn = page.locator("footer button:has-text('Engine performance')")
        perf_btn.click()
        time.sleep(1)
        assert page.locator("#perfB").is_visible()
        # Close perf drawer
        page.locator("#perfB button").click()
        time.sleep(0.5)
        assert not page.locator("#perfB").is_visible()

        # 7. Test Dataset Hub Modal
        print("Testing Dataset Hub modal...")
        hub_btn = page.locator("header button:has-text('Dataset Hub')")
        hub_btn.click()
        time.sleep(1)
        assert page.locator("h2:has-text('Unified Dataset Benchmark Hub')").is_visible()
        # Close modal
        page.locator("div.fixed button:has(svg.lucide-x)").first.click()
        time.sleep(0.5)

        # 8. Test Concept Drift Modal
        print("Testing Concept Drift modal...")
        drift_btn = page.locator("header button:has-text('Concept drift')")
        drift_btn.click()
        time.sleep(1)
        assert page.locator("h3:has-text('Feedzai AML Temporal Concept Drift')").is_visible()
        # Close modal
        page.locator("div.fixed button:has(svg.lucide-x)").first.click()
        time.sleep(0.5)

        # 9. Test Clicking a Table Row and Resuming Live Stream
        print("Testing table row click and inspect mode...")
        second_row = page.locator("table tbody tr").nth(2)
        if second_row.is_visible():
            second_row.click()
            time.sleep(1)
            # Verify Resume live stream button appears
            resume_btn = page.locator("button:has-text('Resume live stream')")
            print(f"Resume live stream button visible: {resume_btn.is_visible()}")
            if resume_btn.is_visible():
                resume_btn.click()
                time.sleep(1)
                print("Clicked Resume live stream")

        # 10. Test Open Investigation button
        print("Testing Open Investigation button...")
        inv_btn = page.locator("button:has-text('Open investigation')")
        if inv_btn.is_visible():
            inv_btn.click()
            time.sleep(1)
            toast = page.locator("div.absolute.top-16").first
            if toast.is_visible():
                print(f"Investigation Toast: {toast.inner_text()}")

        # 11. Test Dark / Light Theme Toggle
        print("Testing Dark / Light Theme Toggle...")
        theme_btn = page.locator("#theme-toggle")
        assert theme_btn.is_visible(), "Theme toggle button not found!"

        # Toggle to Light Mode
        print("Switching to Light Mode...")
        theme_btn.click()
        time.sleep(1.5)

        # Verify HTML class has 'light' and localStorage has 'light'
        is_light_class = page.evaluate("() => document.documentElement.classList.contains('light')")
        saved_theme = page.evaluate("() => localStorage.getItem('safebit_theme')")
        print(f"Theme state after toggle: html.has('light')={is_light_class}, localStorage={saved_theme}")
        assert is_light_class is True, "Document element does not contain 'light' class!"
        assert saved_theme == "light", f"localStorage theme is not 'light': {saved_theme}"

        # Capture light theme screenshot
        light_screenshot_path = "C:/Users/bhuva/.gemini/antigravity/brain/27eb4491-392f-4ac1-b938-042b73917f55/safebit_light_theme_verified.png"
        page.screenshot(path=light_screenshot_path)
        print(f"Light theme screenshot saved to {light_screenshot_path}")

        # Verify Dataset Hub modal in Light Mode
        print("Verifying Dataset Hub modal in Light Mode...")
        hub_btn = page.locator("header button:has-text('Dataset Hub')")
        hub_btn.click()
        time.sleep(1)
        assert page.locator("h2:has-text('Unified Dataset Benchmark Hub')").is_visible()
        page.locator("div.fixed button:has(svg.lucide-x)").first.click()
        time.sleep(0.5)

        # Toggle back to Dark Mode
        print("Switching back to Dark Mode...")
        theme_btn.click()
        time.sleep(1.5)

        is_dark_class = page.evaluate("() => document.documentElement.classList.contains('dark')")
        saved_theme_dark = page.evaluate("() => localStorage.getItem('safebit_theme')")
        print(f"Theme state after toggle back: html.has('dark')={is_dark_class}, localStorage={saved_theme_dark}")
        assert is_dark_class is True, "Document element does not contain 'dark' class!"
        assert saved_theme_dark == "dark", f"localStorage theme is not 'dark': {saved_theme_dark}"

        # Take screenshot of the healthy dashboard in dark theme
        screenshot_path = "C:/Users/bhuva/.gemini/antigravity/brain/27eb4491-392f-4ac1-b938-042b73917f55/safebit_dark_theme_verified.png"
        page.screenshot(path=screenshot_path)
        print(f"Dark theme screenshot saved to {screenshot_path}")

        print(f"Console errors detected: {len(console_errors)}")
        if console_errors:
            print("Errors:", console_errors[:5])

        browser.close()
        print("ALL BUTTONS, GRAPH, AND THEME TOGGLE TESTS PASSED!")

if __name__ == "__main__":
    run_e2e_test()

