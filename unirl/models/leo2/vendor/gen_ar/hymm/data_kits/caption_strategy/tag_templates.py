style_templates_zh = [
    "这张图片采用了{}风格。",
    "这是一幅典型的{}作品。",
    "画面呈现出鲜明的{}特征。",
    "这张图片呈现出{}风格",
    "整幅作品体现了{}的美学。",
    "这是一张充满{}风格的图片。",
    "画面展示了{}的特点。",
    "这张图是{}风格的。",
    "是一幅{}风格的作品。",
    "整体是{}的风格表现。",
    "这是一幅典型的{}风格图片。",
    "这张{}风格的图很有特色。",
    "画面带有明显的{}风格。",
    "这是{}风格。",
    "图片运用了{}的表现手法。",
    "{}风格在这张图中很突出。",
    "这张图完美呈现了{}风格。",
    "很标准的{}风格作品。",
    "图片体现了{}的典型特征。",
    "{}风格让这张图很特别。",
    "这张图的{}风格很明显。",
    "画面展现了{}的独特风格。",
    "这是张很有代表性的{}风格图。",
    "采用{}风格。",
    "图片采用了纯正的{}风格。",
    "典型的{}风格表现手法。",
    "这张{}风格的图很有味道。",
    "画面是标准的{}风格。",
    "有明显的{}风格。",
    "呈现出{}风格的手法。",
    "图片展现了{}风格。",
    "这是张很典型的{}风格图。",
    "整张图都是{}的风格特点。",
]

style_templates_en = [
    "This image adopts a {} style.",
    "This is a typical {} piece.",
    "The composition displays distinct {} characteristics.",
    "This image presents a {} style.",
    "The artist employs {} techniques, infusing the work with expressive power.",
    "The entire piece embodies {} aesthetics.",
    "This is an image rich in {} charm, with a pronounced style.",
    "The visuals interpret the essence of {}.",
    "This image showcases the allure of {}, with a highly distinctive style.",
    "The {} approach makes this piece visually striking.",
    "This is a {} style image.",
    "It's a {} style artwork.",
    "The overall presentation reflects {} styling.",
    "This is a quintessential {} style picture.",
    "This {} style image has remarkable character.",
    "The visuals carry unmistakable {} stylization.",
    "This is {} style.",
    "The image utilizes {} techniques.",
    "The {} style stands out prominently here.",
    "This image perfectly captures {} style.",
    "A textbook {} style creation.",
    "The image demonstrates {}'s hallmark features.",
    "The {} treatment makes this piece distinctive.",
    "The {} styling is evident here.",
    "The composition exhibits {}'s unique approach.",
    "This is a representative {} style image.",
    "The {} styling is executed masterfully.",
    "The image employs authentic {} styling.",
    "A classic {} style rendering technique.",
    "This {} style image has exceptional appeal.",
    "The visuals represent standard {} styling.",
    "The {} characteristics are clearly visible.",
    "Presented through {} style methodology.",
    "The image manifests {} styling.",
    "This is a genuine {} style image.",
    "The entire composition reflects {}'s stylistic traits.",
]

style_templates = {
    "zh": style_templates_zh,
    "en": style_templates_en,
}


style_v2_zh2en = {
    "黑白摄影": [
        "Black and White Photography", "Monochrome Photography", "B&W Photography", "Grayscale Photography"
    ],
    "赛博朋克风格": [
        "Cyberpunk", "Cyberpunk Style", "Cyberpunk Aesthetic", "Sci-Fi Cyberpunk"
    ],
    "赛博朋克": [
        "Cyberpunk", "Cyberpunk Style", "Cyberpunk Aesthetic", "Sci-Fi Cyberpunk"
    ],
    "蒸汽朋克": [
        "Steampunk", "Steampunk Style",
    ],
    "黑白动漫/漫画": {
        "zh": [
            "黑白动漫", "黑白漫画"
        ],
        "en": [
            "Black and White Anime", "Black and White Manga", "Monochrome Anime", "Monochrome Manga", "B&W Anime", "B&W Manga"
        ],
    },
    "漫画": [
        "Manga", "Comic", "Comics"
    ],
    "波普艺术": [
        "Pop Art", "Pop Art Style", "Pop Culture Art", "Andy Warhol Style"
    ],
    "美漫": [
        "American Comics", "Western Comics", "US Comic Style", "Comic Book Style", "Superhero Comic Style"
    ],
    "迪士尼风格/皮克斯风格": {
        "zh": [
            "迪士尼风格", "皮克斯风格"
        ],
        "en": [
            "Disney Style", "Pixar Style", "Disney Animation Style", "Pixar Animation Style", "Disney-Pixar Style"
        ],
    },
    "长曝光": [
        "Long Exposure", "Long Exposure Photography", "Slow Shutter Photography", "Light Trail Photography"
    ],
    "低多边形风格": [
        "Low Poly Style", "Low Polygon Style", "Low Poly Art", "Polygonal Art"
    ],
    "老照片风格": [
        "Vintage Photo Style", "Old Photo Style", "Retro Photography", "Antique Photo Style", "Sepia Photography"
    ],
    "复古": [
        "Vintage",
    ],
    "卡通风格": [
        "Cartoon", "Cartoon Style",
    ],
    "卡通": [
        "Cartoon", "Cartoon Style",
    ],
    "摄影与摄像风格": {
        "zh": [
            "摄影风格", "摄影与摄像风格"
        ],
        "en": [
            "Photography Style", "Photographic Style",
        ],
    },
    "传统绘画": [
        "Traditional Painting", "Classic Painting", "Conventional Painting", "Fine Art Painting"
    ],
    "抽象画": [
        "Abstract Art", "Abstract Painting", "Abstract Style", "Non-figurative Art"
    ],
    "线条绘画风格": [
        "Line Art Style", "Line Drawing", "Line Illustration", "Outline Art"
    ],
    "儿童涂鸦": [
        "Children's Doodle", "Kids' Doodle", "Childlike Doodle", "Child's Scribble"
    ],
    "3D风格": [
        "3D Style", "Three-Dimensional Style", "3D Art", "3D Illustration", "3D Rendering",
    ],
    "3d风格": [
        "3D Style", "Three-Dimensional Style", "3D Art", "3D Illustration", "3D Rendering",
    ],
    "废土风格": [
        "Post-Apocalyptic Style", "Wasteland Style", "Apocalyptic Style", "Dystopian Style", "Ruined World Style"
    ],
    "设计": [
        "Design", "Design Style", "Designer Style", "Graphic Design"
    ],
    "涂鸦风格": [
        "Graffiti Style", "Graffiti Art", "Street Art Style", "Urban Graffiti"
    ],
    "数字插画": [
        "Digital Illustration", "Digital Art", "Digital Drawing", "Digital Painting"
    ],
    "日漫": [
        "Japanese Anime", "Japanese Manga", "Anime Style", "Manga Style", "J-Anime", "J-Manga"
    ],
    "超现实主义": [
        "Surrealism", "Surrealist Art", "Surrealistic Style", "Dreamlike Art"
    ],
    "立体主义": [
        "Cubism", "Cubist Art", "Cubist Style", "Geometric Abstraction"
    ],
    "像素风": [
        "Pixel Art", "Pixel Style", "8-bit Art", "Retro Pixel Art"
    ],
    "像素艺术": [
        "Pixel Art", "Pixel Style", "8-bit Art", "Retro Pixel Art"
    ],
    "儿童风格": [
        "Childlike Style", "Kids' Style", "Children's Art Style", "Naive Art"
    ],
    "动漫风格": [
        "Anime", "Anime Style", "Manga Style", "Japanese Animation Style", "Anime-Inspired Style"
    ],
    "动漫": [
        "Anime", "Anime Style", "Manga Style", "Japanese Animation Style", "Anime-Inspired Style"
    ],
    "现代绘画": [
        "Modern Painting", "Modern Art", "Contemporary Painting", "Modernist Art"
    ],
    "世界地图": [
        "World Map", "Global Map", "Map of the World", "Earth Map"
    ],
    "Q版": [
        "Chibi", "Chibi Style",
    ],
    "古典油画": [
        "Classical Oil Painting",
    ],
    "油画": [
        "Oil Painting",
    ],
    "数字艺术": [
        "Digital Art",
    ],
    "哥特": [
        "Gothic",
    ],
    "手绘": [
        "Hand-drawn",
    ],
    "印象派": [
        "Impressionist",
    ],
    "水墨速写": [
        "Ink Sketch",
    ],
    "魔幻现实主义": [
        "Magical Realism",
    ],
    "文艺复兴": [
        "Renaissance",
    ],
    "复古未来": [
        "Retro-futuristic",
    ],
    "超写实": [
        "Hyperrealism", "Hyper-realism", "Super Realism",
    ],
    "超现实": [
        "Surreal",
    ],
    "矢量插画": [
        "Vector Illustration",
    ],
    "水彩": [
        "Watercolor",
    ]
}
