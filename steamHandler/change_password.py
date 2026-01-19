"""Steam password change using Selenium - AUTO-STEAM-RENT style."""
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def change_steam_password(username: str, old_password: str, new_password: str, two_factor_code: str = None) -> bool:
    """
    Change Steam password using Selenium browser automation.
    AUTO-STEAM-RENT style - this is the reliable method they use.
    """
    driver = None
    try:
        logging.info("Starting Selenium password change for account: %s", username)
        
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
        driver.find_element(By.NAME, "password").send_keys(old_password)
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
        
        # Step 4: Navigate to change password page
        driver.get("https://store.steampowered.com/account/changepassword")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.NAME, "password"))
        )
        
        # Step 5: Fill password change form
        driver.find_element(By.NAME, "password").send_keys(old_password)
        driver.find_element(By.NAME, "newpassword").send_keys(new_password)
        driver.find_element(By.NAME, "renewpassword").send_keys(new_password)
        driver.find_element(By.CSS_SELECTOR, ".btn_green_steamui.btn_medium").click()
        
        # Step 6: Check for success
        try:
            WebDriverWait(driver, 15).until(
                lambda d: "changepassword_finish" in d.current_url or "successfully" in d.page_source.lower()
            )
            if "successfully changed" in driver.page_source.lower() or "changepassword_finish" in driver.current_url:
                logging.info("Password changed successfully for account %s", username)
                return True
            else:
                logging.warning("Password change page did not confirm success for account %s", username)
                return False
        except Exception as e:
            logging.warning("Failed to confirm password change success for account %s: %s", username, e)
            # Check if we're on the finish page anyway
            if "changepassword_finish" in driver.current_url:
                return True
            return False
        
    except Exception as e:
        logging.exception("Selenium password change failed for account %s: %s", username, e)
        return False
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
