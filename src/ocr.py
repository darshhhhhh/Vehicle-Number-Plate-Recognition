import re
import cv2
import easyocr


class PlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)

        self.valid_state_codes = {
            "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
            "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA",
            "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB",
            "PY", "RJ", "SK", "TN", "TS", "TR", "UK", "UP", "WB"
        }

    def character_distance(self, a, b):
        """
        Gives lower penalty for visually similar OCR mistakes.
        """
        similar_pairs = {
            ("0", "O"), ("O", "0"),
            ("0", "D"), ("D", "0"),
            ("O", "D"), ("D", "O"),
            ("8", "B"), ("B", "8"),
            ("6", "G"), ("G", "6"),
            ("1", "I"), ("I", "1"),
            ("1", "L"), ("L", "1"),
            ("5", "S"), ("S", "5"),
            ("2", "Z"), ("Z", "2"),
        }

        if a == b:
            return 0

        if (a, b) in similar_pairs:
            return 0.25

        return 1

    def correct_state_code(self, state_code):
        """
        Correct first two characters using valid Indian state codes.
        """
        if len(state_code) != 2:
            return state_code

        state_code = state_code.upper()

        # First apply letter-position fixes
        state_code = state_code.replace("0", "O")
        state_code = state_code.replace("1", "I")
        state_code = state_code.replace("5", "S")
        state_code = state_code.replace("6", "G")
        state_code = state_code.replace("8", "B")

        if state_code in self.valid_state_codes:
            return state_code

        best_code = state_code
        best_score = 999

        for valid_code in self.valid_state_codes:
            score = (
                self.character_distance(state_code[0], valid_code[0])
                + self.character_distance(state_code[1], valid_code[1])
            )

            if score < best_score:
                best_score = score
                best_code = valid_code

        # Only correct if very close
        if best_score <= 0.5:
            return best_code

        return state_code

    def clean_text(self, text):
        """
        Clean OCR output using Indian plate format.

        Standard format:
        LLNNLLNNNN
        Example:
        KA01MN4259
        """
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)

        if len(text) >= 10:
            text = text[:10]
            chars = list(text)

            to_letter = {
                "0": "O",
                "1": "I",
                "5": "S",
                "6": "G",
                "8": "B"
            }

            to_number = {
                "O": "0",
                "D": "0",
                "Q": "0",
                "I": "1",
                "L": "1",
                "T": "1",
                "Z": "2",
                "S": "5",
                "G": "6",
                "B": "8"
            }

            # First two characters: state code
            for i in [0, 1]:
                if chars[i] in to_letter:
                    chars[i] = to_letter[chars[i]]

            state_code = "".join(chars[:2])
            corrected_state = self.correct_state_code(state_code)
            chars[0], chars[1] = corrected_state[0], corrected_state[1]

            # Positions 2 and 3: RTO number
            for i in [2, 3]:
                if chars[i] in to_number:
                    chars[i] = to_number[chars[i]]

            # Positions 4 and 5: series letters
            # Important: do not force O ↔ D here.
            # Both O and D can be valid letters.
            for i in [4, 5]:
                if chars[i] in to_letter:
                    chars[i] = to_letter[chars[i]]

            # Positions 6 to 9: vehicle number
            for i in [6, 7, 8, 9]:
                if chars[i] in to_number:
                    chars[i] = to_number[chars[i]]

            text = "".join(chars)

        return text

    def preprocess_plate(self, plate_image):
        processed_images = []

        processed_images.append(plate_image)

        resized = cv2.resize(
            plate_image,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )
        processed_images.append(resized)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        processed_images.append(gray)

        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        processed_images.append(denoised)

        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2
        )
        processed_images.append(thresh)

        _, otsu = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        processed_images.append(otsu)

        return processed_images

    def read_plate(self, plate_image):
        processed_images = self.preprocess_plate(plate_image)

        best_text = ""
        best_confidence = 0.0

        for img in processed_images:
            results = self.reader.readtext(img)

            for result in results:
                detected_text = result[1]
                confidence = result[2]

                cleaned_text = self.clean_text(detected_text)

                if len(cleaned_text) >= 4 and confidence > best_confidence:
                    best_text = cleaned_text
                    best_confidence = confidence

        return best_text, best_confidence