import platform

def choose_palette():
    """
    Return a dict of color settings based on the current OS.
    """
    base = {
        "bg":        "#FFFEEB",  # light cream
        "fg":        "#202020",  # dark grey text
        "button_bg": "#000000",  # black buttons
        "button_fg": "#FFFEEB",  # cream text
        "entry_bg":  "#FFFFFF",  # white inputs
        "entry_fg":  "#202020",
        "red":       "#8D2E2E",
        "blue":      "#5595C0",
        "green":     "#8CD5A7",
    }
    os_name = platform.system()
    if os_name == "Darwin":  # macOS
        override = {
            "button_bg": "#444444",  # higher-contrast dark grey
            "entry_bg":  "#F0F0F0",
        }
    elif os_name == "Windows":
        override = {
            # keep base defaults for Windows
        }
    else:
        override = {}

    palette = {**base, **override}
    return palette
