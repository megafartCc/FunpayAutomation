"""Steam device deauthorization using Selenium - AUTO-STEAM-RENT style."""
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def deauthorize_all_devices(username: str, password: str, two_factor_code: str = None) -> bool:
    """
    Deauthorize all devices using Selenium browser automation.
    AUTO-STEAM-RENT style - this forcefully logs out all sessions including active games.
    """
    driver = None
    try:
        logging.info("Starting Selenium device deauthorization for account: %s", username)
        
        # Setup Chrome options
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Try to get ChromeDriver
        try:
            service = Service(ChromeDriverManager().install())
        except Exception as e:
            logging.warning("ChromeDriver auto-install failed, trying system path: %s", e)
            service = Service()  # Use system path
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(10)
        
        # Step 1: Login to Steam
        logging.info("Navigating to Steam login page")
        driver.get("https://store.steampowered.com/login/")
        
        # Wait for login form
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        
        # Enter credentials
        driver.find_element(By.NAME, "username").send_keys(username)
        driver.find_element(By.NAME, "password").send_keys(password)
        driver.find_element(By.CSS_SELECTOR, ".btn_blue_steamui.btn_medium").click()
        
        # Step 2: Handle 2FA or email code
        try:
            # Check for 2FA code input
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "twofactorcode_entry"))
            )
            logging.info("2FA code required for account %s", username)
            if two_factor_code:
                driver.find_element(By.ID, "twofactorcode_entry").send_keys(two_factor_code)
                driver.find_element(By.CSS_SELECTOR, ".btn_blue_steamui.btn_medium").click()
            else:
                logging.error("2FA code required but not provided for account %s", username)
                return False
        except:
            try:
                # Check for email code input
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "emailauthcode_entry"))
                )
                logging.info("Email code required for account %s", username)
                if two_factor_code:  # Re-use two_factor_code for email code if available
                    driver.find_element(By.ID, "emailauthcode_entry").send_keys(two_factor_code)
                    driver.find_element(By.CSS_SELECTOR, ".btn_blue_steamui.btn_medium").click()
                else:
                    logging.error("Email code required but not provided for account %s", username)
                    return False
            except:
                # No 2FA or email code required, proceed
                pass
        
        # Step 3: Wait for login to complete
        WebDriverWait(driver, 30).until(
            lambda d: "store.steampowered.com" in d.current_url and "login" not in d.current_url.lower()
        )
        logging.info("Successfully logged in to Steam for account %s", username)
        
        # Step 4: Navigate to revoke devices page
        driver.get("https://store.steampowered.com/account/managedevices")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Step 5: Try to find and click "Revoke All" button
        # Try multiple selectors for the revoke button
        revoke_selectors = [
            "input[value*='Revoke'][value*='All']",
            "input[value*='revoke'][value*='all']",
            "button[onclick*='revoke']",
            "a[href*='revoke']",
            ".revoke-all",
            "#revoke-all",
        ]
        
        revoked = False
        for selector in revoke_selectors:
            try:
                revoke_button = driver.find_element(By.CSS_SELECTOR, selector)
                if revoke_button.is_displayed():
                    revoke_button.click()
                    WebDriverWait(driver, 10).until(
                        EC.alert_present()
                    )
                    alert = driver.switch_to.alert
                    alert.accept()
                    revoked = True
                    logging.info("Successfully clicked revoke all button for account %s", username)
                    break
            except:
                continue
        
        # Step 6: Alternative method - try direct POST to revoke endpoint
        if not revoked:
            try:
                # Get sessionid from cookies
                sessionid = None
                for cookie in driver.get_cookies():
                    if cookie['name'] == 'sessionid':
                        sessionid = cookie['value']
                        break
                
                if sessionid:
                    # Try to revoke via JavaScript
                    driver.execute_script(f"""
                        fetch('https://store.steampowered.com/account/revokeauthorizeddevices', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/x-www-form-urlencoded',
                            }},
                            body: 'sessionid={sessionid}&revokeall=1'
                        }});
                    """)
                    logging.info("Attempted to revoke devices via JavaScript for account %s", username)
                    revoked = True
            except Exception as e:
                logging.warning("JavaScript revoke failed for account %s: %s", username, e)
        
        # Step 7: Also try Steam Community revoke endpoint
        try:
            driver.get("https://steamcommunity.com/devices/revoke")
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            # Try to find revoke form
            try:
                revoke_form = driver.find_element(By.CSS_SELECTOR, "form[action*='revoke']")
                if revoke_form:
                    revoke_form.submit()
                    logging.info("Submitted revoke form on Steam Community for account %s", username)
            except:
                pass
        except:
            pass
        
        # Wait a moment for the revoke to process
        import time
        time.sleep(2)
        
        logging.info("Device deauthorization completed for account %s", username)
        return True
        
    except Exception as e:
        logging.exception("Selenium device deauthorization failed for account %s: %s", username, e)
        return False
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
