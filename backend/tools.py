"""
Lenskart Claire AI — Tool Implementations
Face analysis and lens recommendation use curated mock data.
Frame/product search delegates to the in-memory ProductDB.
"""
import random
import hashlib
import json

# ─────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────

MOCK_FRAMES = [
    {
        "id": "LK-HC-001", "name": "Horizon Classic", "brand": "Vincent Chase",
        "style": "wayfarer", "shape_suitability": ["oval", "heart", "diamond", "oblong"],
        "color": "Matte Black", "color_hex": "#1a1a1a",
        "price": 1499, "original_price": 2999,
        "material": "Premium Acetate", "gender": "unisex",
        "rating": 4.5, "reviews": 3124, "frame_width": "medium",
        "bestseller": True, "new_arrival": False,
        "tags": ["bestseller", "classic", "lightweight"],
        "description": "Timeless wayfarer silhouette crafted from premium Italian acetate. The matte finish adds a modern edge to a classic shape."
    },
    {
        "id": "LK-AV-002", "name": "AeroGlide Aviator", "brand": "John Jacobs",
        "style": "aviator", "shape_suitability": ["oval", "round", "heart"],
        "color": "Gold / Green", "color_hex": "#c9a84c",
        "price": 2299, "original_price": 4599,
        "material": "Titanium", "gender": "unisex",
        "rating": 4.7, "reviews": 1842, "frame_width": "medium",
        "bestseller": True, "new_arrival": False,
        "tags": ["premium", "lightweight", "titanium"],
        "description": "Featherlight titanium aviators with anti-reflective green lenses. Built to last a lifetime."
    },
    {
        "id": "LK-RD-003", "name": "Velvet Round", "brand": "Vincent Chase",
        "style": "round", "shape_suitability": ["square", "diamond", "oblong"],
        "color": "Tortoise Brown", "color_hex": "#8B4513",
        "price": 999, "original_price": 1999,
        "material": "Acetate", "gender": "unisex",
        "rating": 4.3, "reviews": 2651, "frame_width": "medium",
        "bestseller": False, "new_arrival": False,
        "tags": ["vintage", "retro", "trendy"],
        "description": "Vintage-inspired round frames with a rich tortoise shell finish. A nod to classic Parisian style."
    },
    {
        "id": "LK-CE-004", "name": "Noir Cat Eye", "brand": "Lenskart Blu",
        "style": "cat-eye", "shape_suitability": ["round", "oval", "square"],
        "color": "Glossy Black", "color_hex": "#0d0d0d",
        "price": 1799, "original_price": 2999,
        "material": "Acetate", "gender": "female",
        "rating": 4.6, "reviews": 1923, "frame_width": "medium",
        "bestseller": False, "new_arrival": True,
        "tags": ["feminine", "editorial", "bold"],
        "description": "Dramatically upswept cat-eye for editorial-worthy style. Unmistakably confident."
    },
    {
        "id": "LK-RM-005", "name": "Clarity Rimless", "brand": "John Jacobs",
        "style": "rimless", "shape_suitability": ["oval", "square", "oblong", "heart"],
        "color": "Silver", "color_hex": "#C0C0C0",
        "price": 2799, "original_price": 4999,
        "material": "Titanium", "gender": "unisex",
        "rating": 4.4, "reviews": 987, "frame_width": "narrow",
        "bestseller": False, "new_arrival": False,
        "tags": ["minimalist", "professional", "lightweight"],
        "description": "Barely-there rimless titanium frames for a clean, professional look. Lightweight at just 8g."
    },
    {
        "id": "LK-GE-006", "name": "Edge Geometric", "brand": "Lenskart Blu",
        "style": "geometric", "shape_suitability": ["oval", "round", "heart"],
        "color": "Rose Gold", "color_hex": "#B76E79",
        "price": 1599, "original_price": 2999,
        "material": "Metal", "gender": "female",
        "rating": 4.5, "reviews": 1445, "frame_width": "medium",
        "bestseller": False, "new_arrival": True,
        "tags": ["trendy", "geometric", "fashion-forward"],
        "description": "Hexagonal rose-gold frames where architectural design meets feminine fashion."
    },
    {
        "id": "LK-SP-007", "name": "ProSport Wrap", "brand": "Lenskart Sports",
        "style": "sports", "shape_suitability": ["oval", "square", "oblong"],
        "color": "Matte Blue", "color_hex": "#1e3a5f",
        "price": 1299, "original_price": 2499,
        "material": "TR90 Nylon", "gender": "male",
        "rating": 4.4, "reviews": 2213, "frame_width": "wide",
        "bestseller": False, "new_arrival": False,
        "tags": ["sports", "durable", "active"],
        "description": "Flexible TR90 wrap-around frames engineered for active lifestyles. Impact resistant."
    },
    {
        "id": "LK-SQ-008", "name": "Urban Square", "brand": "Vincent Chase",
        "style": "square", "shape_suitability": ["oval", "round", "heart"],
        "color": "Crystal Clear", "color_hex": "#d4eaf7",
        "price": 1199, "original_price": 1999,
        "material": "Acetate", "gender": "unisex",
        "rating": 4.2, "reviews": 3321, "frame_width": "medium",
        "bestseller": False, "new_arrival": False,
        "tags": ["minimal", "clear", "contemporary"],
        "description": "Angular clear-frame acetate glasses — minimal aesthetic with maximum impact."
    },
    {
        "id": "LK-BF-009", "name": "Luxe Butterfly", "brand": "Lenskart Gold",
        "style": "butterfly", "shape_suitability": ["square", "diamond", "oblong"],
        "color": "Purple Gradient", "color_hex": "#6a0dad",
        "price": 3299, "original_price": 5999,
        "material": "Premium Acetate", "gender": "female",
        "rating": 4.8, "reviews": 765, "frame_width": "wide",
        "bestseller": True, "new_arrival": True,
        "tags": ["luxury", "statement", "gradient"],
        "description": "Oversized butterfly frames with a mesmerizing purple gradient. Pure statement luxury."
    },
    {
        "id": "LK-OV-010", "name": "Classic Oval", "brand": "Vincent Chase",
        "style": "oval", "shape_suitability": ["square", "diamond", "heart"],
        "color": "Gunmetal", "color_hex": "#2c3e50",
        "price": 899, "original_price": 1799,
        "material": "Metal", "gender": "unisex",
        "rating": 4.1, "reviews": 4231, "frame_width": "narrow",
        "bestseller": False, "new_arrival": False,
        "tags": ["evergreen", "affordable", "versatile"],
        "description": "Sleek metal oval frames that pair effortlessly with any outfit or occasion."
    },
    {
        "id": "LK-HC-011", "name": "Slim Hexagon", "brand": "John Jacobs",
        "style": "geometric", "shape_suitability": ["round", "oval", "square"],
        "color": "Matte Gold", "color_hex": "#b8860b",
        "price": 2199, "original_price": 3999,
        "material": "Titanium", "gender": "unisex",
        "rating": 4.6, "reviews": 1102, "frame_width": "medium",
        "bestseller": False, "new_arrival": False,
        "tags": ["premium", "editorial", "lightweight"],
        "description": "Ultra-thin hexagonal titanium frames for the architecturally conscious."
    },
    {
        "id": "LK-WF-012", "name": "Heritage Clubmaster", "brand": "Lenskart Vintage",
        "style": "clubmaster", "shape_suitability": ["oval", "square", "heart", "diamond"],
        "color": "Black / Gold", "color_hex": "#1a1a1a",
        "price": 1899, "original_price": 3499,
        "material": "Acetate + Metal", "gender": "unisex",
        "rating": 4.7, "reviews": 2876, "frame_width": "medium",
        "bestseller": True, "new_arrival": False,
        "tags": ["heritage", "retro", "iconic"],
        "description": "The iconic clubmaster silhouette — acetate brow bar over a metal lower frame. Timeless."
    },
]

MOCK_LENSES = [
    {
        "id": "LENS-BLU", "name": "BluBlock Pro",
        "tagline": "Shield Your Eyes from Digital Strain",
        "price": 699, "original_price": 1299,
        "description": "Advanced blue-light blocking coating that reduces digital eye strain by up to 85%.",
        "best_for": ["screen_time_high", "zero_power", "single_vision"],
        "features": ["Blue light blocking", "Anti-glare", "UV 400 protection", "Scratch resistant"],
        "thickness": "Standard 1.56", "coating": "BluBlock + AR", "badge": "Most Popular",
        "screen_time_min": 4
    },
    {
        "id": "LENS-SV", "name": "ClearVision Single",
        "tagline": "Crystal-Clear Single Vision",
        "price": 499, "original_price": 999,
        "description": "Premium 1.56 index single vision lenses with industry-leading anti-reflective coating.",
        "best_for": ["single_vision", "mild_prescription"],
        "features": ["Anti-reflective", "UV protection", "Scratch resistant", "Lightweight"],
        "thickness": "Standard 1.56", "coating": "Anti-Reflective", "badge": None,
        "screen_time_min": 0
    },
    {
        "id": "LENS-THIN", "name": "HiIndex Thin",
        "tagline": "Power Without the Bulk",
        "price": 1299, "original_price": 2499,
        "description": "High-index 1.67 lenses for strong prescriptions — up to 50% thinner than standard.",
        "best_for": ["single_vision", "high_prescription"],
        "features": ["Ultra-thin 1.67 index", "Anti-reflective", "UV 400", "Impact resistant"],
        "thickness": "Thin 1.67", "coating": "Premium AR", "badge": "Best for High Power",
        "screen_time_min": 0
    },
    {
        "id": "LENS-PROG", "name": "Progressive Elite",
        "tagline": "See All Distances, Seamlessly",
        "price": 2499, "original_price": 4999,
        "description": "Premium progressive multifocal lenses with wide corridors for effortless switching between distances.",
        "best_for": ["progressive", "bifocal", "age_40_plus"],
        "features": ["Wide corridors", "No-line bifocal", "Anti-reflective", "UV 400"],
        "thickness": "1.60 Index", "coating": "Elite AR", "badge": "Premium",
        "screen_time_min": 0
    },
    {
        "id": "LENS-PHOTO", "name": "Transitions Smart",
        "tagline": "Adapt to Every Light",
        "price": 1799, "original_price": 3299,
        "description": "Photochromic lenses that auto-darken in sunlight and clear indoors. Perfect for outdoor lifestyles.",
        "best_for": ["outdoor", "single_vision", "progressive"],
        "features": ["Auto-tint", "UV 400 protection", "Anti-reflective", "Scratch resistant"],
        "thickness": "1.56 Standard", "coating": "Photochromic", "badge": "Smart Tech",
        "screen_time_min": 0
    },
    {
        "id": "LENS-ZERO", "name": "StyleZero Non-Rx",
        "tagline": "Fashion-Forward Zero Power",
        "price": 299, "original_price": 599,
        "description": "Premium zero-power lenses with anti-reflective coating for fashion eyewear. Pure style, zero power.",
        "best_for": ["zero_power", "fashion"],
        "features": ["Zero power", "Anti-glare", "UV protection", "Crystal clear"],
        "thickness": "Standard", "coating": "Anti-Reflective", "badge": "Fashion",
        "screen_time_min": 0
    },
]

FACE_SHAPES = {
    "oval": {
        "shape": "Oval",
        "description": "You have a beautifully balanced oval face — slightly longer than wide with gently rounded features. The most versatile face shape for frames!",
        "width": "Medium",
        "recommended_styles": ["wayfarer", "aviator", "square", "round", "geometric", "clubmaster"],
        "avoid_styles": [],
        "key_feature": "Balanced proportions — almost any frame style works beautifully",
        "celebrity_match": "Bella Hadid, Ryan Reynolds"
    },
    "round": {
        "shape": "Round",
        "description": "Your face has soft, curved features with nearly equal width and height. Angular frames create a striking contrast that adds definition.",
        "width": "Medium",
        "recommended_styles": ["square", "wayfarer", "geometric", "clubmaster", "rimless"],
        "avoid_styles": ["round", "small frames"],
        "key_feature": "Angular frames add length and definition to soft features",
        "celebrity_match": "Emma Stone, Channing Tatum"
    },
    "square": {
        "shape": "Square",
        "description": "Your face has a strong, defined jawline with angular features and roughly equal width from forehead to jaw. Softer, rounder frames balance those sharp angles beautifully.",
        "width": "Wide",
        "recommended_styles": ["round", "oval", "aviator", "cat-eye", "rimless"],
        "avoid_styles": ["square", "geometric", "boxy frames"],
        "key_feature": "Curved frames soften and balance strong angular features",
        "celebrity_match": "Angelina Jolie, Brad Pitt"
    },
    "heart": {
        "shape": "Heart",
        "description": "Your face is wider at the forehead tapering down to a delicate chin — a beautiful shape! Bottom-heavy or light frames draw attention down to balance proportions.",
        "width": "Wide forehead",
        "recommended_styles": ["aviator", "oval", "rimless", "light frames", "round"],
        "avoid_styles": ["cat-eye", "top-heavy decorative frames"],
        "key_feature": "Light, bottom-heavy frames complement a wide forehead beautifully",
        "celebrity_match": "Reese Witherspoon, Ryan Gosling"
    },
    "diamond": {
        "shape": "Diamond",
        "description": "You have striking high cheekbones with a narrow forehead and jawline — one of the rarest face shapes! Frames that add width at the temple or brow work perfectly.",
        "width": "Narrow",
        "recommended_styles": ["cat-eye", "oval", "rimless", "butterfly", "clubmaster"],
        "avoid_styles": ["narrow frames", "geometric"],
        "key_feature": "Wider frames at the top balance and complement high cheekbones",
        "celebrity_match": "Halle Berry, Johnny Depp"
    },
    "oblong": {
        "shape": "Oblong",
        "description": "Your face is longer than it is wide with a long, straight cheek line. Wider, shorter frames help shorten the appearance of your face and add fullness.",
        "width": "Narrow",
        "recommended_styles": ["round", "square", "butterfly", "oversized", "wayfarer"],
        "avoid_styles": ["narrow rimless", "small frames"],
        "key_feature": "Wide, decorative frames add width and break up a long face shape",
        "celebrity_match": "Sarah Jessica Parker, Adam Driver"
    }
}


# ─────────────────────────────────────────────
# TOOL FUNCTIONS
# ─────────────────────────────────────────────

def analyze_face(image_url: str) -> dict:
    """
    Mock face analysis from photo. Deterministically assigns a face shape
    based on a hash of the image URL for consistent results.
    """
    # Deterministic shape from URL hash
    shapes = list(FACE_SHAPES.keys())
    hash_val = int(hashlib.md5(image_url.encode()).hexdigest(), 16)
    shape_key = shapes[hash_val % len(shapes)]
    face_info = FACE_SHAPES[shape_key]

    # Deterministic face measurements
    widths = ["Narrow (125-130mm)", "Medium (130-140mm)", "Wide (140-148mm)"]
    width = widths[hash_val % 3]
    bridge_sizes = ["15mm", "17mm", "18mm", "20mm"]
    bridge = bridge_sizes[(hash_val // 3) % 4]
    temple_lengths = ["135mm", "140mm", "145mm", "150mm"]
    temple = temple_lengths[(hash_val // 7) % 4]

    return {
        "success": True,
        "shape": face_info["shape"],
        "description": face_info["description"],
        "face_width": width,
        "bridge_size": bridge,
        "temple_length": temple,
        "recommended_styles": face_info["recommended_styles"],
        "avoid_styles": face_info["avoid_styles"],
        "key_feature": face_info["key_feature"],
        "celebrity_match": face_info["celebrity_match"],
        "confidence": round(85 + (hash_val % 13), 1)  # 85–97% confidence
    }


def search_frames(style: str = None, shape: str = None, color: str = None,
                  price_range: str = None, gender: str = None,
                  limit: int = 6) -> dict:
    """
    Legacy frame search — delegates to product_db with translated filter params.
    Kept for backwards compatibility with existing demo/agent code.
    """
    from product_db import get_db
    db = get_db()

    filters: dict = {}
    if shape:
        filters["face_shape"] = shape.lower()
    if style:
        filters["frame_shape"] = style.lower()
    if gender:
        filters["gender"] = gender.lower()
    if color:
        # pass as soft tag
        filters["tags"] = [color.lower()]

    # Parse price_range string → price_max
    if price_range:
        pr = price_range.replace(",", "").replace("₹", "").strip()
        if "-" in pr:
            parts = pr.split("-")
            try:
                filters["price_min"] = float(parts[0].strip())
                filters["price_max"] = float(parts[1].strip())
            except ValueError:
                pass
        elif "under" in pr.lower():
            try:
                filters["price_max"] = float("".join(c for c in pr if c.isdigit()))
            except ValueError:
                pass

    result = db.search(filters, limit=limit)
    # Map product_db response to legacy shape for existing callers
    frames = result.get("results", [])
    # Add legacy fields the frontend expects if absent
    for f in frames:
        f.setdefault("shape_suitability", f.get("face_shape_recommendation", []))
        f.setdefault("rating", f.get("avg_rating", 4.0))
        f.setdefault("reviews", 100 + int(f.get("avg_rating", 4.0) * 200))
        f.setdefault("original_price", f.get("strikeout_price", f.get("price", 0)))
        f.setdefault("bestseller", "bestseller" in f.get("tags", []))
        f.setdefault("new_arrival", "new-arrival" in f.get("tags", []))
        f.setdefault("frame_width", "medium")
        f.setdefault("brand", f.get("brand", "Lenskart"))

    return {
        "success": True,
        "no_match": result.get("no_match", False),
        "total_found": result.get("total_found", len(frames)),
        "frames": frames,
        "applied_filters": result.get("applied_filters", {}),
    }


def search_products(quiz_tags: dict = None, face_shape: str = None,
                    gender: str = None, age: str = None,
                    filters: dict = None) -> dict:
    """
    Smart product search using quiz tags + face shape.
    Returns top-3 results (or no_match).

    quiz_tags: output from gemini/bedrock analyze_quiz_response()["tags"]
    filters: raw ProductDB filter dict (overrides quiz_tags if both given)
    """
    from product_db import get_db
    db = get_db()

    if filters:
        result = db.search(filters, limit=3)
    else:
        result = db.search_by_quiz_tags(
            quiz_tags or {},
            face_shape=face_shape,
            gender=gender,
            age=age,
        )

    products = result.get("results", [])

    # Enrich products with legacy-compat fields
    for p in products:
        p.setdefault("shape_suitability", p.get("face_shape_recommendation", []))
        p.setdefault("rating", p.get("avg_rating", 4.0))
        p.setdefault("reviews", 100 + int(p.get("avg_rating", 4.0) * 200))
        p.setdefault("original_price", p.get("strikeout_price", p.get("price", 0)))
        p.setdefault("bestseller", "bestseller" in p.get("tags", []))
        p.setdefault("new_arrival", "new-arrival" in p.get("tags", []))
        p.setdefault("frame_width", "medium")
        p.setdefault("style", p.get("frame_shape", ""))

    return {
        "success":    True,
        "no_match":   result.get("no_match", False),
        "reason":     result.get("reason", ""),
        "total_found": result.get("total_found", len(products)),
        "products":   products,            # top 3
        "frames":     products,            # alias for carousel compat
        "applied_filters": result.get("applied_filters", {}),
    }


def get_lens_recommendation(prescription_type: str,
                             screen_time_hours: float = 6.0) -> dict:
    """
    Mock lens recommendation based on prescription type and screen usage.
    """
    prescription_type = prescription_type.lower().replace(" ", "_")
    screen_time_hours = float(screen_time_hours) if screen_time_hours else 6.0

    recommendations = []
    reasoning = []

    # Primary recommendation based on prescription
    if prescription_type in ["zero_power", "zero power", "none", "fashion"]:
        recommendations.append("LENS-ZERO")
        reasoning.append("Since you don't need vision correction, StyleZero non-Rx lenses are perfect.")
        if screen_time_hours >= 4:
            recommendations.append("LENS-BLU")
            reasoning.append(f"With {screen_time_hours:.0f}h of screen time daily, BluBlock Pro will protect your eyes from digital strain.")

    elif prescription_type in ["single_vision", "single vision", "sv"]:
        recommendations.append("LENS-SV")
        reasoning.append("ClearVision Single lenses provide sharp correction for your prescription.")
        if screen_time_hours >= 4:
            recommendations.append("LENS-BLU")
            reasoning.append("Adding BluBlock coating will significantly reduce eye fatigue from screens.")

    elif prescription_type in ["progressive", "multifocal", "bifocal"]:
        recommendations.append("LENS-PROG")
        reasoning.append("Progressive Elite lenses give you clear vision at all distances without visible lines.")

    # Add thin lenses for moderate/high prescriptions
    if prescription_type in ["single_vision", "single vision"] and screen_time_hours <= 4:
        recommendations.append("LENS-THIN")
        reasoning.append("HiIndex Thin lenses reduce lens thickness for a more attractive appearance.")

    # Add transitions for outdoor enthusiasts
    recommendations.append("LENS-PHOTO")
    reasoning.append("Transitions Smart lenses are great if you move between indoor and outdoor environments.")

    # Filter to unique lens packages
    seen = set()
    unique_recs = []
    for lid in recommendations:
        if lid not in seen:
            seen.add(lid)
            unique_recs.append(lid)

    packages = [l for l in MOCK_LENSES if l["id"] in unique_recs]

    # Sort by recommendation order
    packages.sort(key=lambda p: unique_recs.index(p["id"]))

    return {
        "success": True,
        "prescription_type": prescription_type,
        "screen_time_hours": screen_time_hours,
        "reasoning": reasoning,
        "packages": packages[:3]
    }


def calculate_fit_confidence(frame_id: str, face_shape: str = None,
                             face_width: str = None) -> dict:
    """
    Mock fit confidence calculation. Returns a realistic score with reasoning.
    """
    # Prefer the real ProductDB (500-product Lenskart catalog); fall back to
    # the legacy MOCK_FRAMES for any IDs not yet migrated.
    frame = next((f for f in MOCK_FRAMES if f["id"] == frame_id), None)
    if not frame:
        try:
            from product_db import get_db
            p = get_db().get(frame_id)
        except Exception:
            p = None
        if p:
            frame = {
                "id":                frame_id,
                "name":              p.get("name"),
                "style":             p.get("frame_shape") or p.get("style") or "rectangular",
                "shape_suitability": p.get("face_shape_recommendation") or ["oval"],
                "color":             p.get("color") or "",
                "frame_width":       "medium",   # feed doesn't expose frame width; assume medium
            }

    if not frame:
        return {"success": False, "error": f"Frame {frame_id} not found"}

    # Base score from face-shape match
    base_score = 60

    if face_shape:
        face_shape_lower = face_shape.lower()
        if face_shape_lower in frame["shape_suitability"]:
            # Primary match gets high score
            idx = frame["shape_suitability"].index(face_shape_lower)
            if idx == 0:
                base_score = 92
            elif idx == 1:
                base_score = 87
            else:
                base_score = 82
        else:
            base_score = 64

    # Adjust by frame width vs face
    if face_width:
        width_lower = face_width.lower()
        frame_width = frame["frame_width"]
        if ("wide" in width_lower and frame_width == "wide") or \
           ("narrow" in width_lower and frame_width == "narrow") or \
           ("medium" in width_lower and frame_width == "medium"):
            base_score = min(base_score + 5, 98)
        elif ("wide" in width_lower and frame_width == "narrow") or \
             ("narrow" in width_lower and frame_width == "wide"):
            base_score = max(base_score - 8, 55)

    # Add small deterministic variation
    hash_val = int(hashlib.md5(frame_id.encode()).hexdigest(), 16) % 4
    final_score = base_score + hash_val - 2

    # Build reasoning
    reasons = []
    if final_score >= 88:
        reasons = [
            f"Excellent shape match — {frame['style']} frames are ideal for {face_shape or 'your'} face shape",
            f"Frame width ({frame['frame_width']}) proportionally balanced for your features",
            f"The {frame['color'].lower()} colorway beautifully complements natural skin tones",
            f"Bridge and temple sizing align with your facial measurements"
        ]
        verdict = "Perfect Match"
    elif final_score >= 78:
        reasons = [
            f"Great compatibility — {frame['style']} frames work well with your face shape",
            f"The {frame['color'].lower()} tone is a versatile choice",
            f"Slight width adjustment may optimize the fit",
        ]
        verdict = "Great Fit"
    else:
        reasons = [
            f"Moderate match — style works, though not the top recommendation for your shape",
            f"The {frame['color'].lower()} is a bold choice that may need trying in-store",
        ]
        verdict = "Good Fit"

    return {
        "success": True,
        "frame_id": frame_id,
        "frame_name": frame["name"],
        "score": final_score,
        "verdict": verdict,
        "reasons": reasons,
        "try_on_available": True
    }


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """Dispatch tool calls by name."""
    try:
        if tool_name == "analyze_face":
            return analyze_face(tool_input.get("image_url", ""))
        elif tool_name == "search_frames":
            return search_frames(
                style=tool_input.get("style"),
                shape=tool_input.get("shape"),
                color=tool_input.get("color"),
                price_range=tool_input.get("price_range"),
                gender=tool_input.get("gender"),
                limit=tool_input.get("limit", 6),
            )
        elif tool_name == "search_products":
            return search_products(
                quiz_tags=tool_input.get("quiz_tags"),
                face_shape=tool_input.get("face_shape"),
                gender=tool_input.get("gender"),
                age=tool_input.get("age"),
                filters=tool_input.get("filters"),
            )
        elif tool_name == "get_lens_recommendation":
            return get_lens_recommendation(
                prescription_type=tool_input.get("prescription_type", "single_vision"),
                screen_time_hours=tool_input.get("screen_time_hours", 6),
            )
        elif tool_name == "calculate_fit_confidence":
            return calculate_fit_confidence(
                frame_id=tool_input.get("frame_id", ""),
                face_shape=tool_input.get("face_shape"),
                face_width=tool_input.get("face_width"),
            )
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
