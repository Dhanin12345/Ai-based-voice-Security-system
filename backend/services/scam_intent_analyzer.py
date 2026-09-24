import re
from typing import Dict, Any, List, Optional
from backend.utils.logger import logger

class ScamIntentAnalyzer:
    """
    Context-aware conversation and intent analysis service for detecting social engineering,
    credential harvesting, payment extortion, and scam requests in unknown caller conversations.

    Operates with conservative intent detection:
    - Never flags purely defensive/safety mentions of OTP or money.
    - Uses neutral security risk levels: LOW RISK, MEDIUM RISK, HIGH RISK, REVIEW REQUIRED.
    """

    def __init__(self):
        # 1. Defensive / Safety Patterns (Negative matches to prevent false positives)
        self.safety_patterns = [
            r"never\s+share\s+(?:your\s+)?(?:otp|pin|password|cvv|code)",
            r"don'?t\s+share\s+(?:your\s+)?(?:otp|pin|password|cvv|code)",
            r"do\s+not\s+share\s+(?:your\s+)?(?:otp|pin|password|cvv|code)",
            r"bank\s+(?:never|will\s+not)\s+ask\s+for",
            r"keep\s+(?:your\s+)?(?:otp|pin|password)\s+(?:secret|safe|confidential)",
            r"beware\s+of\s+fraud",
            r"caution:\s+do\s+not\s+disclose",
            r"for\s+your\s+security,\s+do\s+not",
            r"is\s+your\s+secret\s+otp",
            r"this\s+otp\s+is\s+valid\s+for",
            r"otp\s+is\s+\d{4,6}[;,.\s]+don'?t\s+share"
        ]

        # 2. OTP / 2FA Extraction Requests
        self.otp_patterns = [
            r"(?:tell|give|share|read|provide|send|say)\s+(?:me\s+)?(?:the\s+)?(?:otp|one[- ]time\s+password|verification\s+code|auth(?:entication)?\s+code|sms\s+code|digits?)(?:\s+you\s+(?:just\s+)?received)?",
            r"what\s+is\s+(?:the\s+|your\s+)?(?:otp|code|verification\s+code)",
            r"enter\s+(?:the\s+)?otp\s+on\s+call",
            r"read\s+out\s+(?:the\s+)?(?:4|6)[- ]digit\s+code",
            r"confirm\s+(?:the\s+)?otp\s+for\s+verification"
        ]

        # 3. Money Demands
        self.money_patterns = [
            r"(?:send|transfer|pay|deposit|remit)\s+(?:me\s+)?(?:the\s+)?(?:money|amount|funds|cash|rs\.?|inr|₹|\$)",
            r"(?:send|transfer|pay)\s+(?:₹|rs\.?|inr|\$)?\s*\d+(?:,\d+)*(?:\s*(?:rupees|dollars|inr))?",
            r"transfer\s+the\s+amount\s+immediately",
            r"give\s+me\s+(?:the\s+)?money",
            r"urgent(?:ly)?\s+(?:send|transfer)\s+money",
            r"send\s+money\s+to\s+this\s+account",
            r"send\s+money"
        ]

        # 4. Payment Demands
        self.payment_patterns = [
            r"(?:make|send|process|release|submit)\s+(?:the\s+)?payment(?:\s+immediately)?",
            r"send\s+(?:the\s+)?payment(?:\s+immediately)?",
            r"(?:make\s+this\s+|send\s+via\s+)?upi\s+(?:payment|transfer)",
            r"pay\s+now\s+to\s+avoid\s+(?:blocking|penalty|arrest|disconnection)"
        ]

        # 5. Invoice Requests
        self.invoice_patterns = [
            r"(?:send|pay|confirm|clear|forward)\s+(?:me\s+)?(?:the\s+|this\s+)?invoice",
            r"(?:pay|settle)\s+this\s+invoice",
            r"confirm\s+(?:the\s+)?(?:invoice\s+)?payment",
            r"invoice\s+payment\s+due",
            r"send\s+(?:the\s+)?invoice"
        ]

        # 6. Bank Details & Account Details
        self.bank_details_patterns = [
            r"(?:what\s+is|tell\s+me|share|provide|give\s+me)\s+(?:your\s+)?(?:bank\s+details|bank\s+account|account\s+number|ifsc(?:\s+code)?|cif\s+number)",
            r"(?:net|internet)\s+banking\s+(?:username|user\s+id|credentials)",
            r"bank\s+branch\s+and\s+account\s+details"
        ]

        self.account_details_patterns = [
            r"(?:share|send|give|provide)\s+(?:your\s+)?account\s+details",
            r"(?:share|send)\s+bank\s+account\s+details"
        ]

        # 7. Card Details & UPI Requests
        self.card_upi_patterns = [
            r"(?:what\s+is|tell\s+me|share|provide|enter)\s+(?:your\s+)?(?:16[- ]digit|credit\s+card|debit\s+card|card\s+number|upi\s+id|vpa)",
            r"expiry\s+date\s+and\s+card\s+number",
            r"scan\s+this\s+qr\s+code\s+to\s+receive\s+money"
        ]

        # 8. ATM PIN / Password / CVV Requests (Credential Request)
        self.credential_patterns = [
            r"(?:give|tell|share|enter|send)\s+(?:me\s+)?(?:your\s+)?(?:atm\s+pin|mpin|upi\s+pin|password|cvv(?:\s+number)?|security\s+code)",
            r"what\s+is\s+(?:your\s+)?(?:atm\s+pin|mpin|cvv|password)"
        ]

        # 9. Remote Access & Coercion / Urgency
        self.remote_access_patterns = [
            r"(?:install|download|open)\s+(?:anydesk|teamviewer|quicksupport|rustdesk|screen\s+share|remote\s+support)",
            r"give\s+remote\s+access\s+to\s+your\s+(?:phone|mobile|computer|pc)"
        ]

        self.urgency_patterns = [
            r"immediately",
            r"urgently?",
            r"within\s+\d+\s+minutes",
            r"right\s+now",
            r"account\s+will\s+be\s+blocked",
            r"police\s+(?:case|action|arrest)",
            r"legal\s+action",
            r"electricity\s+power\s+cut"
        ]

    @staticmethod
    def redact_sensitive_data(text: str) -> str:
        """
        Redacts actual numeric OTPs, passwords, PINs, and bank account numbers (Section 15).
        """
        if not text:
            return ""
        # Redact OTP numbers e.g. "OTP is 384921" -> "OTP is [REDACTED CODE]"
        cleaned = re.sub(r'(?i)\b(otp|code|pin|password|cvv)\s*(?:is|:|\s|=)\s*(\d{3,8})\b', r'\1 [REDACTED CODE]', text)
        # Redact 10-18 digit account/card numbers
        cleaned = re.sub(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b', '[REDACTED CARD NUMBER]', cleaned)
        cleaned = re.sub(r'(?i)\b(account(?:\s*no|\s*number)?)\s*(?:is|:|\s|=)\s*(\d{8,18})\b', r'\1 [REDACTED ACCOUNT]', cleaned)
        return cleaned

    def analyze_conversation(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Analyzes a conversation snippet or transcript for scam and social engineering indicators.
        Returns normalized risk score, detected intent, category, and context safety flags.
        """
        if not text or not text.strip():
            return {
                "detected_intent": "NONE",
                "risk_category": "NORMAL CONVERSATION",
                "risk_level": "LOW RISK",
                "risk_score": 5,
                "urgency_detected": False,
                "is_safety_context": False,
                "action": "MONITOR",
                "warning_title": None,
                "warning_message": None,
                "recommendation": "Normal conversational dialogue observed. Continue monitoring.",
                "highlighted_spans": [],
                "analyzed_text": ""
            }

        cleaned_text = text.strip()
        lower_text = cleaned_text.lower()

        # Step 1: Check for explicit safety/defensive context
        for pattern in self.safety_patterns:
            match = re.search(pattern, lower_text)
            if match:
                logger.info(f"Safety/protective context detected in conversation: '{match.group(0)}'")
                return {
                    "detected_intent": "SAFETY_NOTIFICATION",
                    "risk_category": "SAFETY CONTEXT",
                    "risk_level": "LOW RISK",
                    "risk_score": 10,
                    "urgency_detected": False,
                    "is_safety_context": True,
                    "action": "ALLOW",
                    "warning_title": "Informational Safety Notice",
                    "warning_message": "Caller is reciting standard protective warnings. No extraction detected.",
                    "recommendation": "Protective guidance context detected. Call verified safe.",
                    "highlighted_spans": [match.group(0)],
                    "analyzed_text": cleaned_text
                }

        # Step 2: Check for Urgency / Coercion signals
        urgency_detected = False
        urgency_matches = []
        for pat in self.urgency_patterns:
            m = re.search(pat, lower_text)
            if m:
                urgency_detected = True
                urgency_matches.append(m.group(0))

        detected_intent = "NONE"
        risk_category = "NORMAL CONVERSATION"
        risk_level = "LOW RISK"
        risk_score = 15
        action = "MONITOR"
        warning_title = None
        warning_message = None
        recommendation = "Standard conversation context."
        highlighted_spans = []

        # Step 3: Priority Check - PIN / CVV / Password (Credential Request)
        for pat in self.credential_patterns:
            m = re.search(pat, lower_text)
            if m:
                detected_intent = "CREDENTIAL_REQUEST"
                risk_category = "CREDENTIAL_REQUEST"
                risk_level = "HIGH RISK"
                risk_score = 98
                action = "WARNING"
                warning_title = "CRITICAL CREDENTIAL REQUEST"
                warning_message = "The caller is actively requesting an ATM PIN, MPIN, CVV, or confidential password."
                recommendation = "NEVER disclose ATM PINs, CVV, or passwords under any circumstances."
                highlighted_spans.append(m.group(0))
                break

        # Step 4: OTP Request
        if detected_intent == "NONE":
            for pat in self.otp_patterns:
                m = re.search(pat, lower_text)
                if m:
                    detected_intent = "OTP_REQUEST"
                    risk_category = "OTP_REQUEST"
                    risk_level = "HIGH RISK"
                    risk_score = 92
                    action = "WARNING"
                    warning_title = "SENSITIVE INFORMATION REQUEST"
                    warning_message = "The caller appears to be requesting an authentication code or OTP."
                    recommendation = "Do not share OTP, PIN, password, CVV, or other confidential verification codes with callers."
                    highlighted_spans.append(m.group(0))
                    break

        # Step 5: Money Request
        if detected_intent == "NONE":
            for pat in self.money_patterns:
                m = re.search(pat, lower_text)
                if m:
                    detected_intent = "MONEY_REQUEST"
                    risk_category = "MONEY_REQUEST"
                    risk_level = "HIGH RISK"
                    risk_score = 90
                    action = "WARNING"
                    warning_title = "PAYMENT / MONEY TRANSFER REQUEST"
                    warning_message = "The caller is requesting an immediate financial transfer or money payment."
                    recommendation = "Never transfer funds or make immediate payments to unknown callers. Verify authenticity via official channels."
                    highlighted_spans.append(m.group(0))
                    break

        # Step 6: Payment Request
        if detected_intent == "NONE":
            for pat in self.payment_patterns:
                m = re.search(pat, lower_text)
                if m:
                    detected_intent = "PAYMENT_REQUEST"
                    risk_category = "PAYMENT_REQUEST"
                    risk_level = "HIGH RISK"
                    risk_score = 88
                    action = "WARNING"
                    warning_title = "PAYMENT DEMAND"
                    warning_message = "The caller is requesting a direct payment or UPI transaction."
                    recommendation = "Do not send payments or scan QR codes requested by unknown callers."
                    highlighted_spans.append(m.group(0))
                    break

        # Step 7: Invoice Request
        if detected_intent == "NONE":
            for pat in self.invoice_patterns:
                m = re.search(pat, lower_text)
                if m:
                    detected_intent = "INVOICE_REQUEST"
                    risk_category = "INVOICE_REQUEST"
                    risk_level = "HIGH RISK" if urgency_detected else "MEDIUM RISK"
                    risk_score = 85 if urgency_detected else 70
                    action = "WARNING" if urgency_detected else "REVIEW"
                    warning_title = "INVOICE PAYMENT REQUEST"
                    warning_message = "The caller is requesting invoice payment or invoice confirmation."
                    recommendation = "Verify invoice details with internal vendor contacts before approving payments."
                    highlighted_spans.append(m.group(0))
                    break

        # Step 8: Bank Details Request
        if detected_intent == "NONE":
            for pat in self.bank_details_patterns:
                m = re.search(pat, lower_text)
                if m:
                    detected_intent = "BANK_DETAILS_REQUEST"
                    risk_category = "BANK_DETAILS_REQUEST"
                    risk_level = "HIGH RISK" if urgency_detected else "MEDIUM RISK"
                    risk_score = 80 if urgency_detected else 65
                    action = "WARNING" if urgency_detected else "REVIEW"
                    warning_title = "FINANCIAL INFORMATION REQUEST"
                    warning_message = "The caller is requesting bank account or banking identification details."
                    recommendation = "Exercise extreme caution before disclosing financial account numbers or branch details."
                    highlighted_spans.append(m.group(0))
                    break

        # Step 9: Account Details Request
        if detected_intent == "NONE":
            for pat in self.account_details_patterns:
                m = re.search(pat, lower_text)
                if m:
                    detected_intent = "ACCOUNT_DETAILS_REQUEST"
                    risk_category = "ACCOUNT_DETAILS_REQUEST"
                    risk_level = "HIGH RISK" if urgency_detected else "MEDIUM RISK"
                    risk_score = 80 if urgency_detected else 65
                    action = "WARNING" if urgency_detected else "REVIEW"
                    warning_title = "ACCOUNT DETAILS DEMAND"
                    warning_message = "The caller is requesting private account or banking credentials."
                    recommendation = "Do not share personal or company account details with unverified callers."
                    highlighted_spans.append(m.group(0))
                    break

        # Step 10: Card / UPI Details
        if detected_intent == "NONE":
            for pat in self.card_upi_patterns:
                m = re.search(pat, lower_text)
                if m:
                    detected_intent = "CREDENTIAL_REQUEST"
                    risk_category = "CREDENTIAL_REQUEST"
                    risk_level = "HIGH RISK"
                    risk_score = 85
                    action = "WARNING"
                    warning_title = "PAYMENT CREDENTIALS REQUEST"
                    warning_message = "The caller is requesting payment card numbers, UPI IDs, or scanning of unknown QR codes."
                    recommendation = "Do not share 16-digit card numbers or enter UPI PINs to 'receive' money."
                    highlighted_spans.append(m.group(0))
                    break

        # Step 11: Remote Access Request
        if detected_intent == "NONE":
            for pat in self.remote_access_patterns:
                m = re.search(pat, lower_text)
                if m:
                    detected_intent = "OTHER_SUSPICIOUS_REQUEST"
                    risk_category = "OTHER_SUSPICIOUS_REQUEST"
                    risk_level = "HIGH RISK"
                    risk_score = 95
                    action = "WARNING"
                    warning_title = "REMOTE ACCESS APPLICATION DEMAND"
                    warning_message = "The caller is demanding installation of remote desktop software (AnyDesk, TeamViewer, QuickSupport)."
                    recommendation = "Do not install remote access applications or grant device control to unknown callers."
                    highlighted_spans.append(m.group(0))
                    break

        # If urgency is present on neutral text, elevate slightly to MEDIUM RISK
        if detected_intent == "NONE" and urgency_detected:
            detected_intent = "URGENT COERCION"
            risk_category = "SOCIAL ENGINEERING SUSPICION"
            risk_level = "MEDIUM RISK"
            risk_score = 55
            action = "REVIEW REQUIRED"
            warning_title = "HIGH URGENCY DIALOGUE"
            warning_message = "Caller is using urgent coercive language or threats."
            recommendation = "Stay calm. Legitimate institutions do not demand immediate compliance on phone calls."
            highlighted_spans.extend(urgency_matches)

        if urgency_detected and risk_level == "HIGH RISK":
            risk_score = min(100, risk_score + 5)
            highlighted_spans.extend(urgency_matches)

        return {
            "detected_intent": detected_intent,
            "risk_category": risk_category,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "urgency_detected": urgency_detected,
            "is_safety_context": False,
            "action": action,
            "warning_title": warning_title,
            "warning_message": warning_message,
            "recommendation": recommendation,
            "highlighted_spans": list(set(highlighted_spans)),
            "analyzed_text": cleaned_text
        }

scam_intent_analyzer = ScamIntentAnalyzer()
