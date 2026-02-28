import mss
import numpy as np
import cv2

try:
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


class ScreenCapture:
    def __init__(self, window_title="Hollow Knight", width=84, height=84):
        self.sct = mss.mss()
        self.width = width
        self.height = height
        self.monitor = self._find_window(window_title)
    
    def _find_window(self, title):
        """Try to find the game window, fall back to full screen."""
        if HAS_WIN32:
            try:
                hwnd = win32gui.FindWindow(None, title)
                if hwnd:
                    rect = win32gui.GetWindowRect(hwnd)
                    return {
                        "left": rect[0],
                        "top": rect[1],
                        "width": rect[2] - rect[0],
                        "height": rect[3] - rect[1]
                    }
            except Exception:
                pass
        
        # Fallback: capture primary monitor
        print(f"[ScreenCapture] Could not find window '{title}', using full screen")
        return self.sct.monitors[1]
    
    def capture(self):
        """Capture a single frame as grayscale 84x84 normalized numpy array.
        
        Returns:
            np.ndarray of shape (height, width), dtype float32, range [0, 1]
        """
        raw = np.array(self.sct.grab(self.monitor))
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (self.width, self.height))
        return resized.astype(np.float32) / 255.0


if __name__ == "__main__":
    # Quick benchmark
    import time
    cap = ScreenCapture()
    
    start = time.time()
    for _ in range(100):
        frame = cap.capture()
    elapsed = time.time() - start
    
    print(f"Frame shape: {frame.shape}")
    print(f"Frame range: [{frame.min():.2f}, {frame.max():.2f}]")
    print(f"Capture speed: {100 / elapsed:.0f} FPS")