import cv2
import pytesseract
from PIL import Image
import re
from collections import Counter
import json
from fuzzywuzzy import fuzz
import numpy as np
from PIL.ExifTags import TAGS
import time
import os
import io

class wayID:
    def __init__(self, image_path, first_name=None, last_name=None, street_address=None, street_city=None, street_state=None, street_zip=None, date_of_birth=None):
        self.image_path = image_path
        self.provided_info = {
            'first_name': first_name.upper() if first_name else None,
            'last_name': last_name.upper() if last_name else None,
            'street_address': street_address.upper() if street_address else None,
            'street_city': street_city.upper() if street_city else None,
            'street_state': street_state.upper() if street_state else None,
            'street_zip': street_zip if street_zip else None,
            'date_of_birth': date_of_birth if date_of_birth else None
        }
        self.fake_indicators = []
        self.quality_metrics = {}
        self.image_quality = 0
        # Enhanced patterns for better name matching
        self.patterns = {
            "state_header": re.compile(r"(NEW YORK|CALIFORNIA|TEXAS|FLORIDA|etc)\s+STATE", re.IGNORECASE),
            "name": [
                # Multiple name patterns to try
                re.compile(r"([A-Z'-]+)[,.\s]+([A-Z'-]+(?:\s+[A-Z'-]+)*)", re.MULTILINE),  # Last, First
                re.compile(r"([A-Z'-]+(?:\s+[A-Z'-]+)*)\s+([A-Z'-]+)", re.MULTILINE),      # First Last
                re.compile(r"([A-Z'-]+)[,.\s]*\n\s*([A-Z'-]+)", re.MULTILINE),             # Split by newline
            ],
            "date_of_birth": [
                re.compile(r"DOB[,.\s:]+(\d{2}/\d{2}/\d{4})"),
                re.compile(r"(\d{2}/\d{2}/\d{4})")  # Fallback to any date format
            ],
            "license_number": re.compile(r"[A-Z0-9]\s*(\d{3}\s*\d{3}\s*\d{3})\s*[A-Z0-9]"),
            "address": [
                re.compile(r"(\d+\s+[A-Z0-9\s]+(?:ST|AVE|RD|BLVD|APT).+?\d{5})"),
                re.compile(r"(\d+[A-Z0-9\s,]+\d{5})")  # More permissive address pattern
            ],
            "expiration": re.compile(r"(?:EXP|EXPIRES?)[,.\s:]+(\d{2}/\d{2}/\d{4})"),
            "issue_date": re.compile(r"(?:ISS|ISSUED)[,.\s:]+(\d{2}/\d{2}/\d{4})"),
            "class": re.compile(r"CLASS[,.\s:]*([A-Z])")
        }
        # State-specific validation rules
        self.state_rules = {
            "AL": {
                "license_format": r"[0-9]{7}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["350", "351", "352", "353", "354", "355", "356", "357", "358", "359", "360", "361", "362", "363", "364", "365", "366", "367", "368", "369"]
            },
            "AK": {
                "license_format": r"[0-9]{7}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["995", "996", "997", "998", "999"]
            },
            "AZ": {
                "license_format": r"[A-Z][0-9]{8}",
                "valid_classes": ["A", "B", "C", "D", "G", "M"],
                "zip_prefix": ["850", "851", "852", "853", "854", "855", "856", "857", "858", "859", "860", "861", "862", "863", "864", "865"]
            },
            "AR": {
                "license_format": r"[0-9]{4,9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["716", "717", "718", "719", "720", "721", "722", "723", "724", "725", "726", "727", "728", "729"]
            },
            "CA": {
                "license_format": r"[A-Z][0-9]{7}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "912", "913", "914", "915", "916", "917", "918", "919", "920", "921", "922", "923", "924", "925", "926", "927", "928", "929", "930", "931", "932", "933", "934", "935", "936", "937", "938", "939", "940", "941", "942", "943", "944", "945", "946", "947", "948", "949", "950", "951", "952", "953", "954", "955", "956", "957", "958", "959", "960", "961"]
            },
            "CO": {
                "license_format": r"[0-9]{9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["800", "801", "802", "803", "804", "805", "806", "807", "808", "809", "810", "811", "812", "813", "814", "815", "816"]
            },
            "CT": {
                "license_format": r"[0-9]{9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["060", "061", "062", "063", "064", "065", "066", "067", "068", "069"]
            },
            "DE": {
                "license_format": r"[0-9]{1,7}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["197", "198", "199"]
            },
            "FL": {
                "license_format": r"[A-Z][0-9]{12}",
                "valid_classes": ["A", "B", "C", "D", "E", "M"],
                "zip_prefix": ["320", "321", "322", "323", "324", "325", "326", "327", "328", "329", "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340", "341", "342", "343", "344", "345", "346", "347", "348", "349"]
            },
            "GA": {
                "license_format": r"[0-9]{7,9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "398", "399"]
            },
            "HI": {
                "license_format": r"[A-Z][0-9]{8}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["967", "968"]
            },
            "ID": {
                "license_format": r"[A-Z]{2}[0-9]{6}[A-Z]",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["832", "833", "834", "835", "836", "837", "838"]
            },
            "IL": {
                "license_format": r"[A-Z][0-9]{11,12}",
                "valid_classes": ["A", "B", "C", "D", "L", "M"],
                "zip_prefix": ["600", "601", "602", "603", "604", "605", "606", "607", "608", "609", "610", "611", "612", "613", "614", "615", "616", "617", "618", "619", "620", "621", "622", "623", "624", "625", "626", "627", "628", "629"]
            },
            "IN": {
                "license_format": r"[0-9]{9,10}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["460", "461", "462", "463", "464", "465", "466", "467", "468", "469", "470", "471", "472", "473", "474", "475", "476", "477", "478", "479"]
            },
            "IA": {
                "license_format": r"[0-9]{9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["500", "501", "502", "503", "504", "505", "506", "507", "508", "509", "510", "511", "512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "523", "524", "525", "526", "527", "528"]
            },
            "KS": {
                "license_format": r"[A-Z][0-9]{2}-[0-9]{2}-[0-9]{4}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["660", "661", "662", "663", "664", "665", "666", "667", "668", "669", "670", "671", "672", "673", "674", "675", "676", "677", "678", "679"]
            },
            "KY": {
                "license_format": r"[A-Z][0-9]{8,9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["400", "401", "402", "403", "404", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "420", "421", "422", "423", "424", "425", "426", "427"]
            },
            "LA": {
                "license_format": r"[0-9]{1,9}",
                "valid_classes": ["A", "B", "C", "D", "E", "M"],
                "zip_prefix": ["700", "701", "702", "703", "704", "705", "706", "707", "708", "709", "710", "711", "712", "713", "714"]
            },
            "ME": {
                "license_format": r"[0-9]{7}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["039", "040", "041", "042", "043", "044", "045", "046", "047", "048", "049"]
            },
            "MD": {
                "license_format": r"[A-Z][0-9]{12}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219"]
            },
            "MA": {
                "license_format": r"S[0-9]{8}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["010", "011", "012", "013", "014", "015", "016", "017", "018", "019", "020", "021", "022", "023", "024", "025", "026", "027"]
            },
            "MI": {
                "license_format": r"[A-Z][0-9]{12}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["480", "481", "482", "483", "484", "485", "486", "487", "488", "489", "490", "491", "492", "493", "494", "495", "496", "497", "498", "499"]
            },
            "MN": {
                "license_format": r"[A-Z][0-9]{12}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["550", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566", "567"]
            },
            "MS": {
                "license_format": r"[0-9]{9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["386", "387", "388", "389", "390", "391", "392", "393", "394", "395", "396", "397"]
            },
            "MO": {
                "license_format": r"[A-Z][0-9]{5,9}",
                "valid_classes": ["A", "B", "C", "D", "E", "F", "M"],
                "zip_prefix": ["630", "631", "632", "633", "634", "635", "636", "637", "638", "639", "640", "641", "642", "643", "644", "645", "646", "647", "648", "649", "650", "651", "652", "653", "654", "655", "656", "657", "658"]
            },
            "MT": {
                "license_format": r"[A-Z][0-9]{8}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["590", "591", "592", "593", "594", "595", "596", "597", "598", "599"]
            },
            "NE": {
                "license_format": r"[A-Z][0-9]{6,8}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["680", "681", "682", "683", "684", "685", "686", "687", "688", "689", "690", "691", "692", "693"]
            },
            "NV": {
                "license_format": r"[0-9]{9,10}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["889", "890", "891", "892", "893", "894", "895", "896", "897", "898"]
            },
            "NH": {
                "license_format": r"[0-9]{2}[A-Z]{3}[0-9]{5}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["030", "031", "032", "033", "034", "035", "036", "037", "038"]
            },
            "NJ": {
                "license_format": r"[A-Z][0-9]{14}",
                "valid_classes": ["A", "B", "C", "D", "E", "M"],
                "zip_prefix": ["070", "071", "072", "073", "074", "075", "076", "077", "078", "079", "080", "081", "082", "083", "084", "085", "086", "087", "088", "089"]
            },
            "NM": {
                "license_format": r"[0-9]{9}",
                "valid_classes": ["A", "B", "C", "D", "E", "M"],
                "zip_prefix": ["870", "871", "872", "873", "874", "875", "876", "877", "878", "879", "880", "881", "882", "883", "884"]
            },
            "NY": {
                "license_format": r"\d{3}\s?\d{3}\s?\d{3}",
                "valid_classes": ["A", "B", "C", "D", "E", "M"],
                "zip_prefix": ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149"]
            },
            "NC": {
                "license_format": r"[0-9]{12}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["270", "271", "272", "273", "274", "275", "276", "277", "278", "279", "280", "281", "282", "283", "284", "285", "286", "287", "288", "289"]
            },
            "ND": {
                "license_format": r"[A-Z]{3}-[0-9]{2}-[0-9]{4}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["580", "581", "582", "583", "584", "585", "586", "587", "588"]
            },
            "OH": {
                "license_format": r"[A-Z][0-9]{7}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "440", "441", "442", "443", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "454", "455", "456", "457", "458"]
            },
            "OK": {
                "license_format": r"[A-Z][0-9]{9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["730", "731", "732", "733", "734", "735", "736", "737", "738", "739", "740", "741", "742", "743", "744", "745", "746", "747", "748", "749"]
            },
            "OR": {
                "license_format": r"[0-9]{1,9}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["970", "971", "972", "973", "974", "975", "976", "977", "978", "979"]
            },
            "PA": {
                "license_format": r"[0-9]{8}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196"]
            },
            "RI": {
                "license_format": r"[0-9]{7}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["028", "029"]
            },
            "SC": {
                "license_format": r"[0-9]{5,11}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["290", "291", "292", "293", "294", "295", "296", "297", "298", "299"]
            },
            "SD": {
                "license_format": r"[0-9]{6,10}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["570", "571", "572", "573", "574", "575", "576", "577"]
            },
            "TN": {
                "license_format": r"[0-9]{7,9}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["370", "371", "372", "373", "374", "375", "376", "377", "378", "379", "380", "381", "382", "383", "384", "385"]
            },
            "TX": {
                "license_format": r"[0-9]{7,8}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["750", "751", "752", "753", "754", "755", "756", "757", "758", "759", "760", "761", "762", "763", "764", "765", "766", "767", "768", "769", "770", "771", "772", "773", "774", "775", "776", "777", "778", "779", "780", "781", "782", "783", "784", "785", "786", "787", "788", "789", "790", "791", "792", "793", "794", "795", "796", "797", "798", "799"]
            },
            "UT": {
                "license_format": r"[0-9]{4,10}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["840", "841", "842", "843", "844", "845", "846", "847"]
            },
            "VT": {
                "license_format": r"[0-9]{8}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["050", "051", "052", "053", "054", "055", "056", "057", "058", "059"]
            },
            "VA": {
                "license_format": r"[A-Z][0-9]{8,11}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["201", "220", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245"]
            },
            "WA": {
                "license_format": r"[A-Z]{7}[0-9]{3}[A-Z]{2}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["980", "981", "982", "983", "984", "985", "986", "987", "988", "989", "990", "991", "992", "993", "994"]
            },
            "WV": {
                "license_format": r"[A-Z][0-9]{6}",
                "valid_classes": ["A", "B", "C", "D", "E", "F", "M"],
                "zip_prefix": ["247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "262", "263", "264", "265", "266", "267", "268"]
            },
            "WI": {
                "license_format": r"[A-Z][0-9]{13}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["530", "531", "532", "533", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "544", "545", "546", "547", "548", "549"]
            },
            "WY": {
                "license_format": r"[0-9]{9,10}",
                "valid_classes": ["A", "B", "C", "M"],
                "zip_prefix": ["820", "821", "822", "823", "824", "825", "826", "827", "828", "829", "830", "831"]
            },
            "DC": {
                "license_format": r"[0-9]{7}",
                "valid_classes": ["A", "B", "C", "D", "M"],
                "zip_prefix": ["200", "202", "203", "204", "205"]
            }
        }
        self.required_fields = {'name', 'date_of_birth', 'license_number', 'address', 'expiration'}
        
    def _preprocess_image(self):
        '''
        Preprocessing focused on strongest differentiators with aggressive scoring for fakes
        '''
        # Read image
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Calculate metrics focusing on key differentiators
        quality_metrics = {
            "resolution_score": max(0, 100 - ((width * height) / (1000 * 1000) * 100)),
            "color_transition": min(100, self._calculate_color_transitions(hsv)),
            "rainbow_effect": min(100, (np.std(hsv[:, :, 0]) / 75) * 100),
            "blur_score": self._calculate_blur_score(gray),
            "saturation_score": min(100, (np.mean(hsv[:, :, 1]) / 255) * 150),
            "digital_artifacts": min(100, (np.std(ycrcb[::8, ::8, :]) / np.std(ycrcb)) * 50),
            "microprint_score": self._analyze_microprint(gray)
        }
        
        # Store metrics for fraud detection
        self.quality_metrics.update(quality_metrics)
        
        # Focus on most reliable indicators
        self.fake_indicators = []
        
        # Primary indicators with stricter thresholds and higher impact
        if quality_metrics["resolution_score"] > 40:  # Lower threshold
            self.fake_indicators.append("Suspicious image resolution")
            self.image_quality += 15  # Add penalty
        if quality_metrics["color_transition"] > 25:  # Lower threshold
            self.fake_indicators.append("Unnatural color transitions")
            self.image_quality += 15  # Add penalty
        if quality_metrics["microprint_score"] > 50:  # Lower threshold
            self.fake_indicators.append("Suspicious microprint patterns")
            self.image_quality += 10  # Add penalty
            
        # Secondary indicators with adjusted thresholds
        if quality_metrics["rainbow_effect"] > 60:  # Lower threshold
            self.fake_indicators.append("Suspicious rainbow/hologram pattern")
            self.image_quality += 10  # Add penalty
        if quality_metrics["saturation_score"] > 45:  # Lower threshold
            self.fake_indicators.append("Excessive color saturation")
            self.image_quality += 10  # Add penalty
        if quality_metrics["digital_artifacts"] > 55:  # Lower threshold
            self.fake_indicators.append("Digital scanning artifacts detected")
            self.image_quality += 10  # Add penalty
            
        # Additional penalty for multiple indicators
        if len(self.fake_indicators) >= 3:
            self.image_quality += 20  # Significant boost for multiple red flags
        
        # Calculate overall score with extreme weights for strongest indicators
        weights = {
            # Primary metrics - extreme weights for strongest differentiators
            "color_transition": 12.0,     # Dramatically increased
            "resolution_score": 10.0,     # Dramatically increased
            "microprint_score": 8.0,      # Dramatically increased
            "digital_artifacts": 4.0,     # Significantly increased
            
            # Secondary metrics - minimal impact
            "rainbow_effect": 0.1,        # Further reduced
            "blur_score": 0.1,           # Minimal
            "saturation_score": 0.1       # Minimal
        }
        
        weighted_scores = [
            quality_metrics[metric] * weight 
            for metric, weight in weights.items()
        ]
        total_weight = sum(weights.values())
        
        # Calculate base score
        base_score = sum(weighted_scores) / total_weight
        
        # Apply aggressive multipliers for obvious fakes
        indicator_count = len(self.fake_indicators)
        if indicator_count >= 4:
            base_score *= 1.5  # 50% boost for many indicators
        elif indicator_count >= 3:
            base_score *= 1.4  # 40% boost for several indicators
        elif indicator_count >= 2:
            base_score *= 1.3  # 30% boost for multiple indicators
        
        # Additional multiplier for extreme individual scores
        extreme_indicators = sum(1 for metric, weight in weights.items() 
                               if weight > 5.0 and quality_metrics[metric] > 75)
        if extreme_indicators >= 2:
            base_score *= 1.2  # Additional 20% boost for very high individual scores
        
        self.image_quality = min(100, base_score)
        
        # Proceed with normal preprocessing for OCR
        contrast = cv2.convertScaleAbs(gray, alpha=1.75, beta=0)
        denoised = cv2.fastNlMeansDenoising(contrast, None, 5, 7, 21)
        _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _extract_text_from_image(self, image):
        '''
        Simplified text extraction with minimal processing
        '''
        custom_config = (
            '--oem 3 '
            '--psm 6 '
            '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\'()-/" '
        )
        
        text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # normalize spaces
        
        return text

    def _validate_dl_text(self, text):
        result = {
            "validation_details": {},
            "text_fraud_score": 0,
            "scoring_factors": [],
            "extracted_data": {},
            "match_scores": {}
        }
        
        # Normalize text for comparison
        text = text.upper()
        print(f"Searching in text: {text}")

        # Validate ZIP code
        if self.provided_info.get('street_zip') and self.provided_info.get('street_state'):
            zip_code = self.provided_info['street_zip']
            state = self.provided_info['street_state']
            state_abbrev = state if len(state) == 2 else self._get_state_abbreviation(state)
            
            # Check if ZIP prefix is valid for state
            if state_abbrev in self.state_rules:
                zip_prefix = zip_code[:3]
                valid_prefixes = self.state_rules[state_abbrev]["zip_prefix"]
                if zip_prefix not in valid_prefixes:
                    result["text_fraud_score"] += 45
                    result["scoring_factors"].append(f"Invalid ZIP code prefix {zip_prefix} for state {state}")
            
            # Check if ZIP appears in text
            if zip_code not in text:
                result["text_fraud_score"] += 45
                result["scoring_factors"].append(f"ZIP code {zip_code} not found in ID text")
            else:
                result["text_fraud_score"] = max(0, result["text_fraud_score"] - 10)
                result["scoring_factors"].append("ZIP code found in ID text")

        # Name validation with improved matching
        if self.provided_info['first_name'] or self.provided_info['last_name']:
            # Direct text search for names
            if self.provided_info['first_name']:
                first_name = self.provided_info['first_name']
                first_name_score = fuzz.partial_ratio(text, first_name)
                result["match_scores"]["first_name"] = first_name_score
                if first_name_score < 80:
                    result["text_fraud_score"] += 40
                    result["scoring_factors"].append(f"First name low match: {first_name_score}%")
            
            if self.provided_info['last_name']:
                last_name = self.provided_info['last_name']
                last_name_score = fuzz.partial_ratio(text, last_name)
                result["match_scores"]["last_name"] = last_name_score
                if last_name_score < 80:
                    result["text_fraud_score"] += 40
                    result["scoring_factors"].append(f"Last name low match: {last_name_score}%")

            # If no good matches found, try pattern matching
            if (result["match_scores"].get("first_name", 0) < 80 or 
                result["match_scores"].get("last_name", 0) < 80):
                for name_pattern in self.patterns['name']:
                    matches = name_pattern.finditer(text)
                    for match in matches:
                        last_name_part, first_name_part = match.groups()
                        
                        if self.provided_info['first_name']:
                            new_score = fuzz.partial_ratio(first_name_part, first_name)
                            if new_score > result["match_scores"].get("first_name", 0):
                                result["match_scores"]["first_name"] = new_score
                                if new_score >= 80:
                                    result["text_fraud_score"] = max(0, result["text_fraud_score"] - 40)
                                    result["scoring_factors"] = [f for f in result["scoring_factors"] 
                                                               if "first name" not in f.lower()]
                        
                        if self.provided_info['last_name']:
                            new_score = fuzz.partial_ratio(last_name_part, last_name)
                            if new_score > result["match_scores"].get("last_name", 0):
                                result["match_scores"]["last_name"] = new_score
                                if new_score >= 80:
                                    result["text_fraud_score"] = max(0, result["text_fraud_score"] - 40)
                                    result["scoring_factors"] = [f for f in result["scoring_factors"] 
                                                               if "last name" not in f.lower()]

        # Check for common fake indicators in text
        fake_indicators = [
            "SAMPLE", "SPECIMEN", "NOT FOR IDENTIFICATION", "VOID", 
            "NON-VALID", "INVALID", "TEST", "DEMO", "EXAMPLE",
            "NOT A VALID", "NOT VALID", "TRAINING", "PRACTICE"
        ]
        for indicator in fake_indicators:
            if indicator in text:
                result["text_fraud_score"] += 50
                result["scoring_factors"].append(f"Found fake indicator: {indicator}")

        # Check expiration date
        exp_match = self.patterns['expiration'].search(text)
        if exp_match:
            exp_date = exp_match.group(1)
            try:
                from datetime import datetime
                expiry = datetime.strptime(exp_date, '%m/%d/%Y')
                current = datetime.now()
                if expiry < current:
                    result["text_fraud_score"] += 40
                    result["scoring_factors"].append("ID is expired")
            except ValueError:
                result["text_fraud_score"] += 30
                result["scoring_factors"].append("Invalid expiration date format")

        return result

    def output(self):
        '''
        Enhanced output with reweighted scoring (20% text, 80% image)
        '''
        processed_image = self._preprocess_image()
        extracted_text = self._extract_text_from_image(processed_image)
        validation_result = self._validate_dl_text(extracted_text)
        
        # Calculate image fraud score (already 0-100, where 0 is good)
        image_fraud_score = self.image_quality
        
        # Add metadata analysis
        metadata_score, metadata_findings = self._analyze_metadata()
        
        # Adjust weights to include metadata
        if metadata_score > 80:
            text_weight = 0.2
            image_weight = 0.6
            metadata_weight = 0.2
        else:
            text_weight = 0.2
            image_weight = 0.7
            metadata_weight = 0.1
            
        total_fraud_score = (
            validation_result["text_fraud_score"] * text_weight + 
            image_fraud_score * image_weight +
            metadata_score * metadata_weight
        )
        
        # More aggressive normalization for clearer separation
        normalized_score = (total_fraud_score / 60) * 75  # Base scaling
        
        # Additional boost for known suspicious patterns
        if len(self.fake_indicators) >= 2:
            normalized_score = min(100, normalized_score + 10)
        
        # Boost score if text validation found issues
        if validation_result["text_fraud_score"] > 90:
            normalized_score = min(100, normalized_score + 15)
            
        # Clamp between 0 and 100
        normalized_score = min(100, max(0, normalized_score))
        
        result = {
            "fraud_score": round(normalized_score, 1),
            "risk_level": "High" if normalized_score >= 75 else "Medium" if normalized_score >= 50 else "Low",
            "component_scores": {
                "text_fraud_score": {
                    "score": round(validation_result["text_fraud_score"], 1),
                    "weight": f"{text_weight*100}%"
                },
                "image_fraud_score": {
                    "score": round(image_fraud_score, 1),
                    "weight": f"{image_weight*100}%"
                },
                "metadata_analysis": {
                    "score": round(metadata_score, 1),
                    "findings": metadata_findings,
                    "weight": f"{metadata_weight*100}%"
                }
            },
            "match_scores": validation_result["match_scores"],
            "scoring_factors": validation_result["scoring_factors"],
            "quality_metrics": {k: f"{v:.1f}%" for k, v in self.quality_metrics.items()},
            "fake_indicators": self.fake_indicators,
            "raw_text": extracted_text
        }
        
        # Update interpretation guide
        result["score_interpretation"] = {
            "all_scores": "0-100 (0 = good/authentic, 100 = bad/potentially fraudulent)",
            "weighting": {
                "text_matching": f"{text_weight*100}% of total score",
                "image_quality": f"{image_weight*100}% of total score",
                "metadata_analysis": f"{metadata_weight*100}% of total score"
            },
            "risk_levels": {
                "Low": "0-49",
                "Medium": "50-74",
                "High": "75-100"
            }
        }
        
        return json.dumps(result, indent=2)

    def _analyze_microprint(self, gray):
        '''
        Enhanced microprint analysis that looks for:
        1. Fine detail patterns at multiple scales
        2. Consistent line spacing in tiny text regions
        3. High-frequency components characteristic of microprint
        '''
        # 1. Multi-scale detail analysis
        kernel_sizes = [3, 5, 7]  # Different scales for detail detection
        detail_scores = []
        
        for size in kernel_sizes:
            # Sharpen to enhance fine details
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) / (size * 2)
            filtered = cv2.filter2D(gray, -1, kernel)
            
            # Look for high-frequency components
            detail = np.abs(filtered - gray)
            detail_score = np.mean(detail[detail > 10])  # Only consider significant details
            detail_scores.append(detail_score if not np.isnan(detail_score) else 0)
        
        # 2. Line pattern analysis (microprint often has very regular patterns)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        line_pattern = np.sum(np.abs(sobel_y) > 30, axis=1)  # Horizontal line detection
        line_spacing = np.diff(line_pattern)
        spacing_consistency = np.std(line_spacing[line_spacing > 0]) if len(line_spacing[line_spacing > 0]) > 0 else 100
        
        # 3. FFT analysis for high-frequency components
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Look at high-frequency components (outer regions of FFT)
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        high_freq_mask = np.zeros_like(magnitude)
        high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 0
        high_freq_mask[0:h, 0:w] = 1
        high_freq_ratio = np.sum(magnitude * high_freq_mask) / np.sum(magnitude)
        
        # Combine scores
        detail_score = np.mean(detail_scores)
        pattern_score = min(100, spacing_consistency)
        freq_score = min(100, high_freq_ratio * 1000)
        
        # Weight the components
        weights = {
            'detail': 0.4,
            'pattern': 0.3,
            'frequency': 0.3
        }
        
        final_score = (
            detail_score * weights['detail'] +
            pattern_score * weights['pattern'] +
            freq_score * weights['frequency']
        )
        
        # Real IDs should have higher scores due to more fine detail
        # Invert the score so higher means more suspicious (consistent with other metrics)
        return min(100, 100 - final_score)

    def _detect_uv_simulation(self, hsv):
        '''Detect attempted UV pattern simulation'''
        unusual_colors = np.sum((hsv[:,:,0] > 150) & (hsv[:,:,1] > 200))
        return min(100, (unusual_colors / (hsv.shape[0] * hsv.shape[1])) * 200)

    def _detect_photo_tampering(self, image):
        '''
        Check for signs of photo manipulation with looser constraints
        '''
        quality_levels = [90, 75, 60]
        diffs = []
        
        for q in quality_levels:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            decoded = cv2.imdecode(encoded, 1)
            diff = np.mean(np.abs(image.astype(float) - decoded.astype(float)))
            diffs.append(diff)
        
        variance = np.std(diffs) / np.mean(diffs)
        
        # Looser scoring
        return min(100, variance * 150)  # Reduced multiplier

    def _enhanced_rainbow_detection(self, hsv):
        '''
        Enhanced hologram detection that better distinguishes real from fake patterns
        '''
        # Split into regions to analyze pattern distribution
        h, w = hsv.shape[:2]
        regions = [
            hsv[0:h//2, 0:w//2],      # Top-left
            hsv[0:h//2, w//2:w],      # Top-right
            hsv[h//2:h, 0:w//2],      # Bottom-left
            hsv[h//2:h, w//2:w]       # Bottom-right
        ]
        
        # Analyze hue patterns in each region
        region_scores = []
        for region in regions:
            # Calculate hue statistics
            hue = region[:,:,0]
            sat = region[:,:,1]
            
            # Only consider areas with sufficient saturation
            mask = sat > 50
            if np.sum(mask) > 0:
                hue_masked = hue[mask]
                
                # Calculate hue variance and distribution
                hue_std = np.std(hue_masked)
                hue_hist = np.histogram(hue_masked, bins=30)[0]
                peak_ratio = np.max(hue_hist) / np.mean(hue_hist) if np.mean(hue_hist) > 0 else 0
                
                # Real holograms tend to have more controlled variance
                region_score = min(100, (hue_std * 0.5 + peak_ratio * 0.5))
                region_scores.append(region_score)
        
        if not region_scores:
            return 0
        
        # Analyze pattern consistency across regions
        region_std = np.std(region_scores)
        avg_score = np.mean(region_scores)
        
        # Real holograms tend to have more consistent patterns
        consistency_factor = min(1.0, region_std / 20)
        
        # Calculate final score
        # Lower score for more consistent patterns (characteristic of real holograms)
        rainbow_score = avg_score * (0.5 + consistency_factor)
        
        return min(100, rainbow_score)

    def _enhanced_color_transitions(self, hsv):
        '''
        Enhanced color transition detection
        '''
        # Calculate gradients in both directions
        gradient_x = cv2.Sobel(hsv[:,:,0], cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(hsv[:,:,0], cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)
        
        # Look for suspicious patterns
        direction_hist = np.histogram(gradient_direction, bins=36)[0]
        direction_peaks = np.sum(direction_hist > np.mean(direction_hist) * 1.5)
        
        # Calculate mean gradient magnitude
        mean_magnitude = np.mean(gradient_magnitude)
        
        # Combine metrics (higher score = more suspicious transitions)
        transition_score = (mean_magnitude * 0.7 + direction_peaks * 0.3)
        
        return min(100, transition_score * 2)

    def _analyze_texture_uniformity(self, gray):
        '''
        Improved texture uniformity analysis
        '''
        # Use Local Binary Pattern (LBP) for better texture analysis
        kernel_size = 3
        local_std = np.zeros_like(gray, dtype=float)
        
        # Calculate local standard deviation with smaller kernel
        for i in range(kernel_size//2, gray.shape[0]-kernel_size//2):
            for j in range(kernel_size//2, gray.shape[1]-kernel_size//2):
                patch = gray[i-kernel_size//2:i+kernel_size//2+1, 
                           j-kernel_size//2:j+kernel_size//2+1]
                local_std[i,j] = np.std(patch)
        
        # Real IDs should have a mix of uniform and detailed areas
        texture_variation = np.std(local_std) / np.mean(local_std)
        
        # Score where too uniform (low variation) or too random (high variation) is bad
        return min(100, abs(texture_variation - 0.5) * 100)

    def _analyze_color_distribution(self, hsv):
        '''
        Improved color distribution analysis
        '''
        # Analyze hue distribution
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
        hue_hist = hue_hist / np.sum(hue_hist)
        
        # Real IDs typically have specific color ranges
        dominant_hues = np.sum(hue_hist > np.mean(hue_hist) * 2)
        
        # Check saturation distribution
        sat_hist = cv2.calcHist([hsv], [1], None, [256], [0,256])
        sat_hist = sat_hist / np.sum(sat_hist)
        
        # Real IDs should have controlled saturation
        sat_score = np.std(sat_hist) / np.mean(sat_hist[sat_hist > 0])
        
        # Combine scores
        return min(100, (dominant_hues / 180 * 50 + sat_score * 50))

    def _analyze_edge_quality(self, gray):
        '''
        Improved edge quality analysis
        '''
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density at different scales
        density1 = np.count_nonzero(edges1) / edges1.size
        density2 = np.count_nonzero(edges2) / edges2.size
        
        # Real IDs should have clear, sharp edges
        edge_ratio = abs(density1 - density2) / max(density1, density2)
        
        # Check edge continuity
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges1, kernel, iterations=1)
        continuity = np.count_nonzero(dilated - edges1) / max(np.count_nonzero(edges1), 1)
        
        # Combine metrics (higher continuity and consistent multi-scale edges are good)
        return min(100, (edge_ratio * 50 + continuity * 50))

    def _detect_cartoon(self, image, gray):
        '''
        Simplified cartoon detection focusing on:
        1. Presence of solid colors
        2. Sharp color boundaries
        3. Limited color palette
        '''
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Check for solid color regions (cartoons have large areas of same color)
        blur = cv2.medianBlur(gray, 5)
        diff = np.abs(gray - blur)
        solid_color_ratio = np.sum(diff < 5) / diff.size  # More strict threshold
        solid_color_score = solid_color_ratio * 100
        
        # 2. Analyze color boundaries
        edges = cv2.Canny(gray, 30, 100)  # Even lower thresholds for cartoon lines
        edge_score = np.count_nonzero(edges) / edges.size * 100
        
        # 3. Count distinct colors (cartoons use fewer colors)
        # Quantize colors more aggressively
        colors = hsv.reshape(-1, 3)
        colors = np.round(colors / 32) * 32  # Quantize to fewer colors
        unique_colors = np.unique(colors, axis=0)
        color_count = len(unique_colors)
        color_score = max(0, 100 - (color_count / 50))  # Fewer colors = higher score
        
        # Calculate final score with heavy emphasis on solid colors and limited palette
        weights = {
            'solid_color': 0.5,    # Increased weight for solid colors
            'edges': 0.2,          # Reduced edge importance
            'color_count': 0.3     # Significant weight for limited colors
        }
        
        cartoon_score = (
            solid_color_score * weights['solid_color'] +
            edge_score * weights['edges'] +
            color_score * weights['color_count']
        )
        
        # Boost score if we detect very cartoon-like characteristics
        if solid_color_ratio > 0.4 and color_count < 100:  # If lots of solid colors and limited palette
            cartoon_score *= 1.5
        
        return min(100, cartoon_score)

    def _prepare_for_ocr(self, gray):
        '''
        Prepare image for OCR
        '''
        contrast = cv2.convertScaleAbs(gray, alpha=1.75, beta=0)
        denoised = cv2.fastNlMeansDenoising(contrast, None, 5, 7, 21)
        _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
        return binary

    def _calculate_blur_score(self, gray):
        """
        Calculate a blur score where:
        - 0-100 scale (0 = likely genuine, 100 = likely fraudulent)
        - Mid-range blur (like real IDs) scores lowest (best)
        - Too sharp or too blurry scores higher (worse)
        """
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Define the acceptable range for blur
        PERFECT_BLUR = 1000  # Center of the acceptable range
        MIN_ACCEPTABLE = 100  # Lower bound of acceptable range
        MAX_ACCEPTABLE = 5000  # Upper bound of acceptable range
        
        # Debug information
        print(f"\nBlur analysis for {os.path.basename(self.image_path)}:")
        print(f"Raw blur variance: {blur_var:.2f}")
        print(f"Acceptable range: {MIN_ACCEPTABLE} - {MAX_ACCEPTABLE}")
        print(f"Optimal blur: {PERFECT_BLUR}")

        # Calculate score (inverted from previous version)
        if blur_var < MIN_ACCEPTABLE:
            # Too blurry - bad
            ratio = blur_var / MIN_ACCEPTABLE
            score = max(0, min(100, 100 - (ratio * 50)))  # Bound between 0-100
        elif blur_var > MAX_ACCEPTABLE:
            # Too sharp - bad
            excess_sharpness = (blur_var - MAX_ACCEPTABLE) / MAX_ACCEPTABLE
            score = max(0, min(100, 50 + (excess_sharpness * 25)))  # Bound between 0-100
        else:
            # Within acceptable range - calculate score based on distance from perfect
            distance_from_perfect = abs(blur_var - PERFECT_BLUR) / (MAX_ACCEPTABLE - MIN_ACCEPTABLE)
            score = max(0, min(100, distance_from_perfect * 50))  # Bound between 0-100

        print(f"Final blur score: {score:.2f}/100 (lower is better)")
        
        return score

    def _check_official_colors(self, hsv):
        '''
        Check if colors match typical official ID patterns
        '''
        # Define expected color ranges for official IDs
        official_hue_ranges = [(0, 20),    # Red range
                             (100, 140),   # Blue range
                             (20, 40)]     # Orange/Brown range
        
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
        hue_hist = hue_hist / np.sum(hue_hist)
        
        # Check if dominant colors fall within official ranges
        official_color_ratio = 0
        for range_start, range_end in official_hue_ranges:
            official_color_ratio += np.sum(hue_hist[range_start:range_end])
        
        return min(100, (1 - official_color_ratio) * 100)

    def _detect_security_features(self, gray):
        '''
        Detect common security features in IDs
        '''
        # Look for fine detail patterns
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        detail = cv2.filter2D(gray, -1, kernel)
        
        # Check for microprint-like patterns
        detail_score = np.std(detail) / np.mean(detail)
        
        # Look for regular patterns (guilloche)
        fourier = np.fft.fft2(gray)
        fourier_shift = np.fft.fftshift(fourier)
        magnitude = np.abs(fourier_shift)
        
        # Check for regular pattern presence
        pattern_score = np.std(magnitude) / np.mean(magnitude)
        
        return min(100, 100 - ((detail_score + pattern_score) * 50))

    def _analyze_text_placement(self, gray):
        '''
        Analyze text placement patterns
        '''
        # Use horizontal projection to find text rows
        horizontal_proj = np.sum(gray < 128, axis=1)
        
        # Find peaks in projection (text lines)
        peaks = []
        threshold = np.mean(horizontal_proj) * 1.5
        for i in range(1, len(horizontal_proj) - 1):
            if horizontal_proj[i] > threshold:
                if horizontal_proj[i] > horizontal_proj[i-1] and horizontal_proj[i] > horizontal_proj[i+1]:
                    peaks.append(i)
        
        if len(peaks) < 2:
            return 100  # Suspicious if we can't find enough text lines
        
        # Calculate spacing between text lines
        spacings = np.diff(peaks)
        spacing_consistency = np.std(spacings) / np.mean(spacings)
        
        return min(100, spacing_consistency * 100)

    def _validate_headshot(self, image):
        """
        Analyzes the headshot/photo region of an ID to detect suspicious characteristics
        Returns a score (0-100, where higher is more suspicious) and list of issues
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initialize face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        issues = []
        score = 0
        
        if len(faces) == 0:
            return 75, ["No face detected in ID"]  # Reduced from 100
        elif len(faces) > 1:
            return 75, ["Multiple faces detected in ID"]  # Reduced from 100
        
        # Analyze the detected face
        x, y, w, h = faces[0]
        face_region = image[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        
        # Calculate metrics
        metrics = {
            "face_size_ratio": (w * h) / (width * height),  # Should be ~10-35% of ID (more lenient)
            "face_position_x": x / width,  # Should be on left side (~0.1-0.45)
            "face_position_y": y / height,  # Should be centered (~0.15-0.85)
            "face_aspect_ratio": w / h,  # Should be roughly 0.55-0.95 (more lenient)
            "face_brightness": np.mean(face_gray),  # Check if photo is too dark/bright
            "face_contrast": np.std(face_gray)  # Check if photo has good contrast
        }
        
        # Validate metrics with more lenient thresholds and lower penalties
        if metrics["face_size_ratio"] < 0.08 or metrics["face_size_ratio"] > 0.35:
            score += 15  # Reduced from 25
            issues.append(f"Unusual face size: {metrics['face_size_ratio']:.2%} of ID")
        
        if metrics["face_position_x"] < 0.08 or metrics["face_position_x"] > 0.45:
            score += 15  # Reduced from 25
            issues.append(f"Unusual face position (x): {metrics['face_position_x']:.2%}")
        
        if metrics["face_position_y"] < 0.15 or metrics["face_position_y"] > 0.85:
            score += 15  # Reduced from 25
            issues.append(f"Unusual face position (y): {metrics['face_position_y']:.2%}")
        
        if metrics["face_aspect_ratio"] < 0.55 or metrics["face_aspect_ratio"] > 0.95:
            score += 15  # Reduced from 25
            issues.append(f"Unusual face aspect ratio: {metrics['face_aspect_ratio']:.2f}")
        
        if metrics["face_brightness"] < 40 or metrics["face_brightness"] > 215:  # More lenient
            score += 10  # Reduced from 15
            issues.append(f"Unusual face brightness: {metrics['face_brightness']:.1f}")
        
        if metrics["face_contrast"] < 25:  # More lenient
            score += 10  # Reduced from 15
            issues.append(f"Low face contrast: {metrics['face_contrast']:.1f}")
        
        # Additional checks for photo tampering (reduced penalties)
        try:
            # Check for unusual edges in photo region
            edges = cv2.Canny(face_region, 100, 200)
            edge_density = np.sum(edges > 0) / (w * h)
            if edge_density > 0.35:  # More lenient
                score += 10  # Reduced from 15
                issues.append("Unusual edge patterns in photo")
            
            # Check for color consistency
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            if np.std(hsv[:,:,1]) < 8:  # More lenient
                score += 10  # Reduced from 15
                issues.append("Suspiciously uniform photo coloring")
            
        except Exception as e:
            print(f"Warning: Additional photo checks failed: {str(e)}")
        
        return min(100, score), issues

    def _analyze_metadata(self):
        findings = []
        score = 0
        
        try:
            # Check file extension - expanded list for phone formats
            valid_extensions = {
                '.jpg', '.jpeg', '.png', '.heic', '.mpo',  # Added .mpo
                '.heif', '.dng', '.raw'  # Other common phone formats
            }
            file_ext = os.path.splitext(self.image_path)[1].lower()
            if file_ext not in valid_extensions:
                score += 25
                findings.append(f"Unusual file extension: {file_ext}")
            
            # Get file stats
            file_stats = os.stat(self.image_path)
            current_time = time.time()
            
            # Check file timestamps - only very recent modifications are suspicious
            time_diffs = {
                'modified': current_time - file_stats.st_mtime,
                'accessed': current_time - file_stats.st_atime
            }
            
            # Only flag if modified in last 5 minutes (suggests active tampering)
            if time_diffs['modified'] < 300:
                score += 10
                findings.append("File modified very recently")
            
            # Read image metadata
            with Image.open(self.image_path) as img:
                try:
                    exif = {
                        TAGS[key]: value
                        for key, value in img._getexif().items()
                        if key in TAGS
                    } if img._getexif() else {}
                    
                    # Check for editing software traces
                    software_tags = ['Software', 'ProcessingSoftware', 'Creator']
                    editing_software = [
                        'photoshop', 'gimp', 'paint', 'lightroom', 
                        'illustrator', 'affinity', 'pixelmator'
                    ]
                    
                    # Add common phone camera software to whitelist
                    phone_software = [
                        'iphone', 'ios', 'android', 'samsung', 'pixel',
                        'camera', 'gcam', 'snapdragon'
                    ]
                    
                    for tag in software_tags:
                        if tag in exif:
                            software = str(exif[tag]).lower()
                            if any(editor in software for editor in editing_software):
                                score += 35
                                findings.append(f"Image edited with {exif[tag]}")
                            elif any(phone in software for phone in phone_software):
                                # Reduce score if it's from a phone camera
                                score = max(0, score - 10)
                                findings.append("Image from phone camera")
                    
                except Exception as e:
                    # Don't penalize for missing EXIF - common with phone photos
                    print(f"Info: No EXIF data found")
                
                # Check image format and compression
                format_name = img.format.upper()
                if format_name not in ['JPEG', 'PNG', 'HEIC', 'MPO', 'DNG', 'RAW']:  # Added MPO and others
                    score += 20
                    findings.append(f"Unusual image format: {format_name}")
                elif format_name in ['MPO', 'HEIC']:  # Common phone formats
                    score = max(0, score - 10)  # Reduce score for phone formats
                
                # Check for multiple save operations (JPEG/JPG)
                if format_name == 'JPEG' or file_ext in ['.jpg', '.jpeg']:
                    try:
                        quality_estimate = self._estimate_jpeg_quality(img)
                        # Only flag very low quality
                        if quality_estimate < 50:
                            score += 15
                            findings.append("Suspiciously low JPEG quality")
                    except Exception as e:
                        print(f"Info: Could not estimate JPEG quality")
                
        except Exception as e:
            score += 15
            findings.append(f"Error analyzing metadata: {str(e)}")
        
        return min(100, score), findings

    def _estimate_jpeg_quality(self, img):
        """
        Estimates the JPEG quality setting
        """
        # Save image to buffer at different quality levels
        qualities = [20, 40, 60, 80, 90, 95, 98, 100]
        sizes = []
        original_size = 0
        
        # Get original file size
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=100)
        original_size = len(buffer.getvalue())
        
        # Get sizes at different quality levels
        for q in qualities:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=q)
            sizes.append(len(buffer.getvalue()))
        
        # Find closest match
        size_diffs = [abs(size - original_size) for size in sizes]
        estimated_quality = qualities[np.argmin(size_diffs)]
        
        return estimated_quality

    def _calculate_color_transitions(self, hsv_image):
        """
        Calculate the number of significant color transitions in the HSV image.
        This can help detect fake IDs that might have unusual color patterns.
        
        Args:
            hsv_image: Image in HSV color space
            
        Returns:
            int: Number of significant color transitions found
        """
        # Extract the hue channel
        hue = hsv_image[:, :, 0]
        
        # Calculate horizontal and vertical gradients
        gradient_x = cv2.Sobel(hue, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(hue, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Count significant transitions (adjust threshold as needed)
        threshold = 30
        transitions = np.sum(gradient_magnitude > threshold)
        
        return int(transitions)