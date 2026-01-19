import os
import time
import uuid
import random
import string
import base64
import hmac
import hashlib
import struct
import mysql.connector
from flask import Flask, jsonify, request
from flask_cors import CORS
from steam.client import SteamClient
from steam.enums import EResult
from apscheduler.schedulers.background import BackgroundScheduler

# --- 1. SETUP & CONFIGURATION ---

app = Flask(__name__)
# Allow your website to talk to this API
CORS(app) 

# Function to connect to your Railway MySQL database
def get_db_connection():
    """Connects to the MySQL database using environment variables."""
    return mysql.connector.connect(
        host=os.getenv('MYSQLHOST'),
        database=os.getenv('MYSQLDATABASE'),
        user=os.getenv('MYSQLUSER'),
        password=os.getenv('MYSQLPASSWORD')
    )

def init_steam_db():
    """Creates the 'steam_accounts' table in your database if it doesn't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS steam_accounts (
            username VARCHAR(100) PRIMARY KEY,
            password VARCHAR(100) NOT NULL,
            shared_secret TEXT NOT NULL,
            is_rented TINYINT DEFAULT 0,
            session_id VARCHAR(36),
            expires_at BIGINT
        )
        """)
        conn.commit()
        conn.close()
        print("✅ Steam accounts table is ready.")
    except Exception as e:
        print(f"❌ DATABASE INIT FAILED: {e}")

# --- 2. CORE STEAM HELPER FUNCTIONS ---

def perform_steam_login(username, password, shared_secret):
    """Logs into Steam and returns an authenticated client object."""
    def _generate_code(secret):
        try:
            secret_bytes = base64.b64decode(secret)
            timestamp = int(time.time()) // 30
            msg = struct.pack(">Q", timestamp)
            hmac_hash = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
            offset = hmac_hash[-1] & 0x0F
            code_int = int.from_bytes(hmac_hash[offset:offset + 4], "big") & 0x7FFFFFFF
            alphabet = "23456789BCDFGHJKMNPQRTVWXY"
            code = ""
            for _ in range(5):
                code += alphabet[code_int % len(alphabet)]
                code_int //= len(alphabet)
            return code
        except:
            return None

    try:
        code = _generate_code(shared_secret)
        if not code:
            print(f"[{username}] Failed to generate 2FA code.")
            return None
        client = SteamClient()
        if client.connect():
            result = client.login(username, password, two_factor_code=code)
            if result == EResult.OK:
                return client
    except Exception as e:
        print(f"[{username}] Exception during login: {e}")
    return None

def deauthorize_all_sessions(client):
    """Kicks all other users off the account."""
    try:
        client.web_logoff()
        return True
    except:
        return False

def change_steam_password(client, old_password, new_password):
    """Changes the password for the logged-in account."""
    try:
        sessionid = client.web_session.cookies.get('sessionid', domain='steamcommunity.com')
        if not sessionid: return False
        form_data = {
            'sessionid': sessionid, 'password': old_password,
            'newpassword': new_password, 'renewpassword': new_password
        }
        response = client.web_session.post(
            'https://store.steampowered.com/account/changepassword_finish',
            data=form_data, headers={'Referer': 'https://store.steampowered.com/account/changepassword'}
        )
        return 'successfully changed' in response.text.lower()
    except:
        return False

# --- 3. THE MASTER RESET WORKFLOW ---

def reset_steam_account(username):
    """The main workflow to take back and reset a rented account."""
    print(f"--- [RESET WORKFLOW STARTED FOR: {username}] ---")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password, shared_secret FROM steam_accounts WHERE username=%s", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"ERROR: Account '{username}' not found.")
        return

    old_password, shared_secret = row
    client = perform_steam_login(username, old_password, shared_secret)

    if not client:
        print(f"FAILED: Login failed for '{username}'.")
        return

    try:
        if deauthorize_all_sessions(client):
            print(f"INFO: Sessions deauthorized for '{username}'.")
            new_password = ''.join(random.choice(string.ascii_letters + string.digits + "!@#$") for _ in range(16))
            if change_steam_password(client, old_password, new_password):
                conn_update = get_db_connection()
                cursor_update = conn_update.cursor()
                cursor_update.execute("UPDATE steam_accounts SET password=%s, is_rented=0, session_id=NULL, expires_at=NULL WHERE username=%s", (new_password, username))
                conn_update.commit()
                conn_update.close()
                print(f"SUCCESS: Account '{username}' has been fully reset.")
            else:
                print(f"FAILED: Password change failed for '{username}'.")
        else:
            print(f"FAILED: Could not deauthorize sessions for '{username}'.")
    finally:
        client.logout()
        print(f"--- [RESET WORKFLOW FINISHED FOR: {username}] ---")

# --- 4. API ENDPOINTS FOR YOUR WEBSITE ---

@app.route('/steam/rent', methods=['GET'])
def steam_rent():
    """Rents out an available Steam account."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, password FROM steam_accounts WHERE is_rented=0 LIMIT 1")
    row = cursor.fetchone()

    if row:
        username, password = row
        session_id = str(uuid.uuid4())
        expires_at = int(time.time() + 7200)  # 2 hours from now
        cursor.execute("UPDATE steam_accounts SET is_rented=1, session_id=%s, expires_at=%s WHERE username=%s", (session_id, expires_at, username))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'username': username, 'password': password, 'session_id': session_id})
    else:
        conn.close()
        return jsonify({'success': False, 'error': 'No accounts available at the moment.'}), 404

@app.route('/steam/code', methods=['POST'])
def steam_code():
    """Provides a 2FA code for an active rental session."""
    data = request.json
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({'success': False, 'error': 'session_id is missing.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT shared_secret FROM steam_accounts WHERE session_id=%s AND is_rented=1 AND expires_at > %s", (session_id, int(time.time())))
    row = cursor.fetchone()

    if row:
        shared_secret = row[0]
        # Generate 2FA code from shared_secret
        def _generate_code(secret):
            try:
                secret_bytes = base64.b64decode(secret)
                timestamp = int(time.time()) // 30
                msg = struct.pack(">Q", timestamp)
                hmac_hash = hmac.new(secret_bytes, msg, hashlib.sha1).digest()
                offset = hmac_hash[-1] & 0x0F
                code_int = int.from_bytes(hmac_hash[offset:offset + 4], "big") & 0x7FFFFFFF
                alphabet = "23456789BCDFGHJKMNPQRTVWXY"
                code = ""
                for _ in range(5):
                    code += alphabet[code_int % len(alphabet)]
                    code_int //= len(alphabet)
                return code
            except:
                return None
        
        code = _generate_code(shared_secret)
        conn.close()
        
        if code:
            return jsonify({'success': True, 'code': code})
        else:
            return jsonify({'success': False, 'error': 'Failed to generate 2FA code.'}), 500
    else:
        conn.close()
        return jsonify({'success': False, 'error': 'Invalid or expired session.'}), 404
        
# --- 5. BACKGROUND SCHEDULER ---

def cleanup_expired_rentals():
    """Finds expired rentals and calls the reset workflow."""
    print("SCHEDULER: Checking for expired rentals...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM steam_accounts WHERE is_rented=1 AND expires_at < %s", (int(time.time()),))
    expired_accounts = cursor.fetchall()
    conn.close()

    if expired_accounts:
        print(f"SCHEDULER: Found {len(expired_accounts)} expired account(s).")
        for (username,) in expired_accounts:
            reset_steam_account(username)
    else:
        print("SCHEDULER: No expired accounts found.")

# --- 6. INITIALIZATION & APP START ---

if __name__ == '__main__':
    # Initialize the database table when the app starts
    init_steam_db()

    # Start the background scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=cleanup_expired_rentals, trigger="interval", seconds=60)
    scheduler.start()
    
    # Run the Flask app
    # The port is dynamically assigned by Railway
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
