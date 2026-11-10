import numpy as np
import cv2
from PIL import Image
import gradio as gr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
import matplotlib.pyplot as plt

# COLOR PALETTES

PALETTES = {
    "Grayscale": np.array([[i, i, i] for i in range(0, 256, 32)]),
    "Ocean": np.array([[0, 30, 60], [0, 60, 100], [0, 100, 140], [0, 140, 180],
                       [30, 170, 200], [80, 200, 220], [140, 220, 235], [200, 240, 250]]),
    "Vibrant": np.array([[255, 0, 68], [255, 119, 0], [255, 238, 0], [0, 255, 68],
                         [0, 238, 255], [0, 119, 255], [68, 0, 255], [255, 0, 238]]),
    "Sunset": np.array([[50, 20, 50], [100, 30, 60], [150, 50, 70], [200, 80, 80],
                        [230, 120, 90], [250, 160, 100], [255, 200, 120], [255, 230, 160]]),
    "Forest": np.array([[34, 139, 34], [46, 139, 87], [60, 179, 113], [144, 238, 144],
                        [85, 107, 47], [128, 128, 0], [152, 251, 152], [173, 255, 47]]),
}

# GRID DIVISION (VECTORIZED)

def divide_into_grid_vectorized(image, grid_size=(32, 32)):
    """
    Vectorized grid division - computes average color per tile.
    grid_size is (num_rows, num_cols) - not pixel size!
    """
    img_h, img_w = image.shape[:2]
    grid_rows, grid_cols = grid_size
    
    # Calculate tile dimensions from image / grid count
    tile_h = img_h // grid_rows
    tile_w = img_w // grid_cols
    
    cropped_h = tile_h * grid_rows
    cropped_w = tile_w * grid_cols
    image_cropped = image[:cropped_h, :cropped_w]
    
    # Reshape for vectorization
    reshaped = image_cropped.reshape(grid_rows, tile_h, grid_cols, tile_w, 3)
    tile_colors = reshaped.mean(axis=(1, 3)).astype(np.uint8)
    
    return tile_colors, (tile_h, tile_w), image_cropped, (grid_rows, grid_cols)

def divide_into_grid_loop(image, grid_size=(32, 32)):
    """Loop-based version for comparison."""
    img_h, img_w = image.shape[:2]
    grid_rows, grid_cols = grid_size
    
    tile_h = img_h // grid_rows
    tile_w = img_w // grid_cols
    
    cropped_h = tile_h * grid_rows
    cropped_w = tile_w * grid_cols
    image_cropped = image[:cropped_h, :cropped_w]
    
    tile_colors = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            y1, y2 = i * tile_h, (i + 1) * tile_h
            x1, x2 = j * tile_w, (j + 1) * tile_w
            tile_colors[i, j] = image_cropped[y1:y2, x1:x2].mean(axis=(0, 1))
    
    return tile_colors, (tile_h, tile_w), image_cropped, (grid_rows, grid_cols)

def quantize_to_palette(tile_colors, palette):
    """Map each tile to nearest palette color."""
    colors_expanded = tile_colors[:, :, np.newaxis, :]
    distances = np.sqrt(((colors_expanded - palette) ** 2).sum(axis=3))
    indices = np.argmin(distances, axis=2)
    quantized = palette[indices]
    return quantized.astype(np.uint8), indices

# TILE GENERATORS

def generate_solid_tiles(palette, tile_size):
    """Solid color tiles."""
    tiles = {}
    for idx, color in enumerate(palette):
        tile = np.full((tile_size[0], tile_size[1], 3), color, dtype=np.uint8)
        tiles[idx] = tile
    return tiles

def generate_gradient_tiles(palette, tile_size):
    """Vertical gradient tiles."""
    tiles = {}
    for idx, color in enumerate(palette):
        tile = np.zeros((tile_size[0], tile_size[1], 3), dtype=np.uint8)
        for i in range(tile_size[0]):
            factor = 0.5 + 0.5 * (i / tile_size[0])
            tile[i, :] = np.clip(color.astype(float) * factor, 0, 255).astype(np.uint8)
        tiles[idx] = tile
    return tiles

def generate_striped_tiles(palette, tile_size):
    """Horizontal striped tiles."""
    tiles = {}
    for idx, color in enumerate(palette):
        tile = np.full((tile_size[0], tile_size[1], 3), color, dtype=np.uint8)
        for i in range(0, tile_size[0], max(1, tile_size[0] // 4)):
            tile[i:i+1, :] = np.clip(color.astype(float) * 0.7, 0, 255).astype(np.uint8)
        tiles[idx] = tile
    return tiles

def generate_circle_tiles(palette, tile_size):
    """Circle pattern tiles."""
    tiles = {}
    h, w = tile_size
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    radius = min(h, w) / 2 - 2
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        tile[mask] = color
        tiles[idx] = tile
    return tiles

def generate_diamond_tiles(palette, tile_size):
    """Diamond pattern tiles."""
    tiles = {}
    h, w = tile_size
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    mask = (np.abs(x - center_x) + np.abs(y - center_y)) <= min(h, w) / 2 - 2
    
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        tile[mask] = color
        tiles[idx] = tile
    return tiles

def generate_triangle_tiles(palette, tile_size):
    """Triangle pattern tiles."""
    tiles = {}
    h, w = tile_size
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        for i in range(h):
            left = int(w / 2 - (i / h) * w / 2)
            right = int(w / 2 + (i / h) * w / 2)
            tile[i, max(0, left):min(w, right)] = color
        tiles[idx] = tile
    return tiles

def generate_crosshatch_tiles(palette, tile_size):
    """Crosshatch pattern tiles."""
    tiles = {}
    h, w = tile_size
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        for i in range(0, h, max(1, h // 3)):
            tile[i:i+1, :] = color
        for j in range(0, w, max(1, w // 3)):
            tile[:, j:j+1] = color
        tiles[idx] = tile
    return tiles

def generate_noise_tiles(palette, tile_size):
    """Noise texture tiles."""
    tiles = {}
    h, w = tile_size
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), color, dtype=np.uint8)
        noise = np.random.randint(-30, 30, (h, w, 3))
        tile = np.clip(tile.astype(int) + noise, 0, 255).astype(np.uint8)
        tiles[idx] = tile
    return tiles

def generate_hexagon_tiles(palette, tile_size):
    """Hexagon pattern tiles."""
    tiles = {}
    h, w = tile_size
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        from PIL import Image as PILImage, ImageDraw
        pil_tile = PILImage.new('RGB', (w, h), tuple(200 for _ in range(3)))
        draw = ImageDraw.Draw(pil_tile)
        
        # Draw hexagon
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 2
        vertices = []
        for i in range(6):
            angle = np.pi / 3 * i
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            vertices.append((x, y))
        
        draw.polygon(vertices, fill=tuple(color), outline=tuple(color))
        tiles[idx] = np.array(pil_tile)
    return tiles

def generate_fibonacci_tiles(palette, tile_size):
    """Fibonacci spiral tiles."""
    tiles = {}
    h, w = tile_size
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        from PIL import Image as PILImage, ImageDraw
        pil_tile = PILImage.new('RGB', (w, h), tuple(200 for _ in range(3)))
        draw = ImageDraw.Draw(pil_tile)
        
        # Draw fibonacci rectangles
        fib = [1, 1, 2, 3, 5, 8]
        max_fib = max(fib)
        scale = min(w, h) / max_fib
        
        px, py = 2, 2
        for f in fib:
            scaled = int(f * scale)
            draw.rectangle([px, py, px + scaled, py + scaled], outline=tuple(color), width=1)
            if fib.index(f) % 2 == 0:
                px += scaled
            else:
                py += scaled
        
        tiles[idx] = np.array(pil_tile)
    return tiles

def generate_golden_ratio_tiles(palette, tile_size):
    """Golden ratio spiral tiles."""
    tiles = {}
    h, w = tile_size
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        from PIL import Image as PILImage, ImageDraw
        pil_tile = PILImage.new('RGB', (w, h), tuple(200 for _ in range(3)))
        draw = ImageDraw.Draw(pil_tile)
        
        # Draw concentric circles with golden ratio spacing
        cx, cy = w // 2, h // 2
        max_r = min(w, h) // 2
        phi = 1.618
        
        r = 2
        while r < max_r:
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=tuple(color), width=1)
            r = int(r * phi)
        
        tiles[idx] = np.array(pil_tile)
    return tiles

def generate_mandelbrot_tiles(palette, tile_size):
    """Mandelbrot-inspired fractal pattern tiles."""
    tiles = {}
    h, w = tile_size
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        
        # Simple iterative pattern inspired by Mandelbrot
        for i in range(h):
            for j in range(w):
                # Map to complex plane
                x = (j - w/2) / (w/4)
                y = (i - h/2) / (h/4)
                
                # Simple iteration
                iterations = 0
                zx, zy = 0, 0
                for _ in range(10):
                    zx, zy = zx*zx - zy*zy + x, 2*zx*zy + y
                    if zx*zx + zy*zy > 4:
                        break
                    iterations += 1
                
                if iterations > 5:
                    tile[i, j] = color
        
        tiles[idx] = tile
    return tiles

def generate_emoji_tiles(palette, tile_size):
    """Emoji-inspired pattern tiles with actual visual patterns."""
    tiles = {}
    h, w = tile_size
    
    emoji_patterns = [
        "star",      # Star pattern
        "smile",     # Smile face
        "circle",    # Circle
        "diamond",   # Diamond
        "square",    # Square
        "cross",     # Cross/plus
        "dot",       # Center dot
        "ring"       # Ring
    ]
    
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        from PIL import Image as PILImage, ImageDraw
        pil_tile = PILImage.new('RGB', (w, h), (200, 200, 200))
        draw = ImageDraw.Draw(pil_tile)
        
        pattern = emoji_patterns[idx % len(emoji_patterns)]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3
        
        if pattern == "star":
            # Draw star
            vertices = []
            for i in range(10):
                angle = np.pi / 5 * i
                radius = r if i % 2 == 0 else r // 2
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                vertices.append((x, y))
            draw.polygon(vertices, fill=tuple(color), outline=tuple(color))
        
        elif pattern == "smile":
            # Draw smiley face
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=tuple(color), width=2)
            draw.ellipse([cx-r//3, cy-r//4, cx-r//6, cy], fill=tuple(color))
            draw.ellipse([cx+r//6, cy-r//4, cx+r//3, cy], fill=tuple(color))
            draw.arc([cx-r//2, cy, cx+r//2, cy+r], 0, 180, fill=tuple(color), width=2)
        
        elif pattern == "circle":
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=tuple(color), outline=tuple(color))
        
        elif pattern == "diamond":
            vertices = [(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)]
            draw.polygon(vertices, fill=tuple(color), outline=tuple(color))
        
        elif pattern == "square":
            draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=tuple(color), outline=tuple(color))
        
        elif pattern == "cross":
            draw.line([(cx, cy-r), (cx, cy+r)], fill=tuple(color), width=2)
            draw.line([(cx-r, cy), (cx+r, cy)], fill=tuple(color), width=2)
        
        elif pattern == "dot":
            draw.ellipse([cx-r//2, cy-r//2, cx+r//2, cy+r//2], fill=tuple(color))
        
        elif pattern == "ring":
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=tuple(color), width=3)
        
        tiles[idx] = np.array(pil_tile)
    return tiles

def generate_rhombus_tiles(palette, tile_size):
    """Rhombus/diamond pattern tiles."""
    tiles = {}
    h, w = tile_size
    for idx, color in enumerate(palette):
        tile = np.full((h, w, 3), 200, dtype=np.uint8)
        from PIL import Image as PILImage, ImageDraw
        pil_tile = PILImage.new('RGB', (w, h), tuple(200 for _ in range(3)))
        draw = ImageDraw.Draw(pil_tile)
        
        # Draw rhombus
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 2
        vertices = [
            (cx, cy - r),      # top
            (cx + r, cy),      # right
            (cx, cy + r),      # bottom
            (cx - r, cy)       # left
        ]
        draw.polygon(vertices, fill=tuple(color), outline=tuple(color))
        tiles[idx] = np.array(pil_tile)
    return tiles

TILE_GENERATORS = {
    "Solid": generate_solid_tiles,
    "Gradient": generate_gradient_tiles,
    "Striped": generate_striped_tiles,
    "Circle": generate_circle_tiles,
    "Diamond": generate_diamond_tiles,
    "Triangle": generate_triangle_tiles,
    "Crosshatch": generate_crosshatch_tiles,
    "Noise": generate_noise_tiles,
    "Hexagon": generate_hexagon_tiles,
    "Fibonacci": generate_fibonacci_tiles,
    "Golden Ratio": generate_golden_ratio_tiles,
    "Mandelbrot": generate_mandelbrot_tiles,
    "Emoji": generate_emoji_tiles,
    "Rhombus": generate_rhombus_tiles,
}

# MOSAIC BUILDING

def build_mosaic(indices, tiles, tile_size, grid_size):
    """Reconstruct mosaic from tile indices."""
    grid_rows, grid_cols = grid_size
    tile_h, tile_w = tile_size
    
    mosaic_h = grid_rows * tile_h
    mosaic_w = grid_cols * tile_w
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            tile_idx = int(indices[i, j])
            y1, y2 = i * tile_h, (i + 1) * tile_h
            x1, x2 = j * tile_w, (j + 1) * tile_w
            mosaic[y1:y2, x1:x2] = tiles[tile_idx]
    
    return mosaic

# METRICS

def calculate_metrics(original, mosaic):
    """Calculate MSE, SSIM, PSNR."""
    if mosaic.shape != original.shape:
        mosaic = cv2.resize(mosaic, (original.shape[1], original.shape[0]))
    
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    
    mse = np.mean((orig_gray.astype(float) - mosaic_gray.astype(float)) ** 2)
    ssim_score = ssim(orig_gray, mosaic_gray)
    psnr_score = psnr(orig_gray, mosaic_gray, data_range=255)
    
    return mse, ssim_score, psnr_score

# MAIN PIPELINE

def create_mosaic(image, grid_size, palette_name, tile_style):
    """Complete mosaic transformation pipeline."""
    if image is None:
        return None, None, None, "No image provided"
    
    try:
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
        else:
            image_bgr = image
        
        grid_size = int(grid_size)
        
        # Step 1: Divide into grid
        tile_colors, tile_dims, cropped_image, grid_dims = divide_into_grid_vectorized(
            image_bgr, (grid_size, grid_size)
        )
        
        # Step 2: Quantize to palette
        # Palettes are defined in RGB, but the pipeline works in OpenCV's BGR order
        palette = PALETTES[palette_name][:, ::-1].copy()
        quantized_colors, indices = quantize_to_palette(tile_colors, palette)
        
        # Step 3: Generate tiles
        tile_generator = TILE_GENERATORS[tile_style]
        tiles = tile_generator(palette, tile_dims)
        
        # Step 4: Build mosaic
        mosaic = build_mosaic(indices, tiles, tile_dims, grid_dims)
        
        # Create segmented view
        segmented = np.repeat(np.repeat(quantized_colors, tile_dims[0], axis=0), 
                              tile_dims[1], axis=1)
        
        # Step 5: Calculate metrics
        mse, ssim_score, psnr_score = calculate_metrics(cropped_image, mosaic)
        
        metrics_text = f"""
📊 PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MSE:  {mse:.2f} (lower is better)
SSIM: {ssim_score:.4f} (higher is better)
PSNR: {psnr_score:.2f} dB (higher is better)

📐 CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Grid: {grid_size}×{grid_size} tiles
Tile size: {tile_dims[0]}×{tile_dims[1]} pixels
Palette: {palette_name}
Tile style: {tile_style}
"""
        
        # Convert back to RGB for display
        original_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        
        return original_rgb, segmented_rgb, mosaic_rgb, metrics_text
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, f"Error: {str(e)}"

# BENCHMARKING

def benchmark_implementations(image, grid_sizes=[8, 16, 32, 64, 128]):
    """Compare vectorized vs loop-based implementations."""
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    results = {
        "grid_size": [],
        "vectorized": [],
        "loop_based": [],
        "speedup": []
    }
    
    for gs in grid_sizes:
        # Vectorized
        start = time.perf_counter()
        divide_into_grid_vectorized(image, (gs, gs))
        vec_time = time.perf_counter() - start
        
        # Loop-based
        start = time.perf_counter()
        divide_into_grid_loop(image, (gs, gs))
        loop_time = time.perf_counter() - start
        
        results["grid_size"].append(f"{gs}×{gs}")
        results["vectorized"].append(vec_time * 1000)
        results["loop_based"].append(loop_time * 1000)
        results["speedup"].append(loop_time / vec_time if vec_time > 0 else 1)
    
    return results

def plot_benchmark(results):
    """Create benchmark visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(results["grid_size"]))
    width = 0.35
    
    # Plot 1: Execution time
    axes[0].bar(x - width/2, results["vectorized"], width, label="Vectorized", color="steelblue")
    axes[0].bar(x + width/2, results["loop_based"], width, label="Loop-based", color="coral")
    axes[0].set_xlabel("Grid Size")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Execution Time Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results["grid_size"], rotation=45)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Speedup
    colors = plt.cm.RdYlGn(np.array(results["speedup"]) / max(results["speedup"]))
    axes[1].bar(x, results["speedup"], color=colors)
    axes[1].axhline(y=1, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Grid Size")
    axes[1].set_ylabel("Speedup (x)")
    axes[1].set_title("Vectorization Speedup")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results["grid_size"], rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(axes[1].patches, results["speedup"])):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def run_benchmark(image):
    """Run benchmark and return results."""
    if image is None:
        return None, "No image provided"
    
    results = benchmark_implementations(image, [8, 16, 32, 64, 128])
    fig = plot_benchmark(results)
    
    text = "VECTORIZED VS LOOP-BASED PERFORMANCE\n"
    text += "=" * 50 + "\n"
    for i in range(len(results["grid_size"])):
        text += f"{results['grid_size'][i]}: Vectorized={results['vectorized'][i]:.2f}ms, "
        text += f"Loop={results['loop_based'][i]:.2f}ms, Speedup={results['speedup'][i]:.1f}x\n"
    
    return fig, text

# GRADIO INTERFACE

with gr.Blocks(title="Image Mosaic Generator") as demo:
    gr.Markdown("# 🎨 Image Mosaic Generator")
    
    with gr.Tab("🖼️ Create Mosaic"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="pil")
                grid_size = gr.Slider(8, 128, 32, 8, label="Grid Size (tiles per side)")
                palette = gr.Dropdown(list(PALETTES.keys()), "Grayscale", label="Color Palette")
                tile_style = gr.Dropdown(list(TILE_GENERATORS.keys()), "Solid", label="Tile Style")
                generate_btn = gr.Button("🎨 Generate", variant="primary")
            
            with gr.Column():
                metrics = gr.Textbox(label="Metrics", interactive=False, lines=8)
        
        with gr.Row():
            original = gr.Image(label="Original")
            segmented = gr.Image(label="Segmented")
            mosaic = gr.Image(label="Mosaic Result")
        
        gr.Markdown("**💡 Tip:** Right-click on any image and select 'Save image as...' to download")
        
        generate_btn.click(
            create_mosaic,
            [image_input, grid_size, palette, tile_style],
            [original, segmented, mosaic, metrics]
        )
    
    with gr.Tab("📊 Performance"):
        bench_image = gr.Image(label="Upload Image", type="pil")
        bench_btn = gr.Button("Run Benchmark", variant="primary")
        bench_plot = gr.Plot()
        bench_text = gr.Textbox(label="Results", interactive=False, lines=8)
        
        bench_btn.click(run_benchmark, [bench_image], [bench_plot, bench_text])

if __name__ == "__main__":
    demo.launch()