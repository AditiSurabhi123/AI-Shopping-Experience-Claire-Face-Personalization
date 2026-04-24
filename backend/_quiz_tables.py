"""
Quiz tables for Claire — each question is one natural conversational line,
with examples baked in so no separate "intro" is needed. This prevents TTS
from saying the same thing twice (intro + question).

Order:
    1 gender · 2 age · 3 product_type · 4 lifestyle ·
    5 style · 6 color · 7 budget · 8 vision
"""

QUIZ_TOTAL = 8

QUIZ_EN = {
    1: {"q": "Are these frames for you, for him, for her, or for everyone?",
        "opts": ["For me", "For him", "For her", "For everyone"],
        "key": "gender",       "intro": ""},

    2: {"q": "And the age — just a number in years, like 10, 20, 35, or 55?",
        "opts": ["Under 13", "Around 20", "Around 35", "50 or older"],
        "key": "age",          "intro": ""},

    3: {"q": "Are we looking at eyeglasses, sunglasses, or contact lenses?",
        "opts": ["Eyeglasses", "Sunglasses", "Contact lenses"],
        "key": "product_type", "intro": ""},

    4: {"q": "What's the daily vibe — office work, something creative, active and outdoors, or fashion-forward?",
        "opts": ["Office Professional", "Creative & Artistic", "Active & Outdoorsy", "Fashion-Forward"],
        "key": "lifestyle",    "intro": ""},

    5: {"q": "Which style feels most like you — classic and timeless, bold and trendy, minimal and clean, or adventurous?",
        "opts": ["Classic & Timeless", "Bold & Trendy", "Minimal & Clean", "Adventurous"],
        "key": "style_pref",   "intro": ""},

    6: {"q": "Which Frame Colour are looking for — neutral blacks and greys, warm golds, cool blues, or a transparent?",
        "opts": ["Neutral (Black/Grey)", "Warm (Gold/Tortoise)", "Cool (Blue/Silver)", "Statement Colour"],
        "key": "color_pref",   "intro": ""},

    7: {"q": "Budget-wise, what suits — under 1000 rupees, 1000 to 2000, 2000 to 3500, or premium above 3500?",
        "opts": ["Under 1000", "1000 to 2000", "2000 to 3500", "Premium above 3500"],
        "key": "budget",       "intro": ""},

    8: {"q": "And the vision — do you have trouble seeing up close, or far away? Maybe it's mainly for computer work, all-round, or just zero power for fashion?",
        "opts": ["Zero power (fashion only)", "Single vision", "Progressive / Bifocal", "Computer / reading glasses"],
        "key": "prescription", "intro": ""},
}

QUIZ_HI = {
    1: {"q": "ये चश्मा आपके लिए है, किसी पुरुष के लिए, किसी महिला के लिए, या सबके लिए?",
        "opts": ["मेरे लिए", "उनके लिए (पुरुष)", "उनके लिए (महिला)", "सबके लिए"],
        "key": "gender",       "intro": ""},

    2: {"q": "आप अपनी उम्र बताइए — जैसे 10 साल, 20 साल, 35 साल, 55 साल",
        "opts": ["13 साल से कम", "करीब 20 साल", "करीब 35 साल", "50 साल या उससे ज़्यादा"],
        "key": "age",          "intro": ""},

    3: {"q": "आपको क्या चाहिए — साधारण चश्मा, धूप का चश्मा, या कॉन्टैक्ट लेंस?",
        "opts": ["साधारण चश्मा", "धूप का चश्मा", "कॉन्टैक्ट लेंस"],
        "key": "product_type", "intro": ""},

    4: {"q": "आपका दिन कैसा गुज़रता है — ऑफिस में काम, क्रिएटिव काम, एक्टिव आउटडोर, या फैशन?",
        "opts": ["ऑफिस प्रोफेशनल", "क्रिएटिव आर्टिस्ट", "एक्टिव स्पोर्ट्स", "फैशन फॉरवर्ड"],
        "key": "lifestyle",    "intro": ""},

    5: {"q": "आपको कौन सा स्टाइल पसंद है — क्लासिक, बोल्ड, मिनिमल, या एडवेंचरस?",
        "opts": ["क्लासिक टाइमलेस", "बोल्ड ट्रेंडी", "मिनिमल क्लीन", "एडवेंचरस"],
        "key": "style_pref",   "intro": ""},

    6: {"q": "कौन सा रंग अच्छा लगेगा — काला और ग्रे, सुनहरा या ब्राउन, नीला या सिल्वर, या कोई चमकीला रंग?",
        "opts": ["न्यूट्रल काला या ग्रे", "वार्म गोल्ड या टॉर्टॉइज़", "कूल ब्लू या सिल्वर", "कोई बोल्ड रंग"],
        "key": "color_pref",   "intro": ""},

    7: {"q": "बजट kitna है aapka — हज़ार से कम, एक से दो हज़ार, दो से साढ़े तीन हज़ार, या साढ़े तीन हज़ार से ज़्यादा?",
        "opts": ["हज़ार से कम", "एक से दो हज़ार", "दो से साढ़े तीन हज़ार", "प्रीमियम साढ़े तीन हज़ार से ज़्यादा"],
        "key": "budget",       "intro": ""},

    8: {"q": "और आख़िरी सवाल — नज़र की ज़रूरत क्या है? पास का नहीं दिखता, दूर का नहीं दिखता, कंप्यूटर के लिए, या सिर्फ स्टाइल के लिए?",
        "opts": ["ज़ीरो पावर (फैशन)", "सिंगल विज़न", "प्रोग्रेसिव / बाइफोकल", "कंप्यूटर / पढ़ने के लिए"],
        "key": "prescription", "intro": ""},
}
