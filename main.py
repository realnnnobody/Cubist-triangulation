import argparse
import cv2
import numpy as np
from scipy.spatial import Delaunay
from numba import njit


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to open file: {path}")
    return img


def lerp(a, b, t):
    return int(a + (b - a) * t)


def ensure_uint8_mask(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask


def foreground_mask(img, use_fg=True, margin=0.06, iters=5):
    h, w = img.shape[:2]

    if not use_fg:
        return np.ones((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), np.uint8)

    mx = int(w * margin)
    my = int(h * margin)
    rect_w = max(1, w - 2 * mx)
    rect_h = max(1, h - 2 * my)
    rect = (mx, my, rect_w, rect_h)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0
        ).astype(np.uint8)
    except cv2.error:
        fg = np.ones((h, w), dtype=np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    if fg.sum() < h * w * 0.03:
        fg = np.ones((h, w), dtype=np.uint8)

    return fg


def border_points(img, step=30):
    h, w = img.shape[:2]
    pts = []

    for x in range(0, w, step):
        pts.append([x, 0])
        pts.append([x, h - 1])

    for y in range(0, h, step):
        pts.append([0, y])
        pts.append([w - 1, y])

    pts.extend([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    return np.array(pts, dtype=np.int32)


def largest_contour_points(mask, step=4):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.int32)

    cnt = max(contours, key=cv2.contourArea)[:, 0, :]
    if len(cnt) <= step:
        return cnt.astype(np.int32)

    return cnt[::step].astype(np.int32)


def edge_points(img, mask=None, canny1=60, canny2=160, step=4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny1, canny2)

    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    pts = np.column_stack([xs, ys])
    pts = pts[::step]
    return pts.astype(np.int32)


def gradient_map(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    score = np.zeros(img.shape[:2], dtype=np.float32)

    for ch in cv2.split(lab):
        sx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        score += cv2.magnitude(sx, sy)

    score = cv2.GaussianBlur(score, (5, 5), 0)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX)
    return score.astype(np.float32)


def transition_points(img, mask=None, block=12, min_score=18.0):
    grad = gradient_map(img)
    h, w = grad.shape
    pts = []

    for y in range(0, h, block):
        for x in range(0, w, block):
            y2 = min(y + block, h)
            x2 = min(x + block, w)
            patch = grad[y:y2, x:x2]
            if patch.size == 0:
                continue

            idx = np.unravel_index(np.argmax(patch), patch.shape)
            py = y + idx[0]
            px = x + idx[1]

            if mask is not None and mask[py, px] == 0:
                continue

            if grad[py, px] >= min_score:
                pts.append([px, py])

    if not pts:
        return np.empty((0, 2), dtype=np.int32)

    return np.array(pts, dtype=np.int32)


def random_interior_points(mask, count=700):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    count = min(count, len(xs))
    idx = np.random.choice(len(xs), size=count, replace=False)
    pts = np.column_stack([xs[idx], ys[idx]])
    return pts.astype(np.int32)


def unique_points(points):
    if len(points) == 0:
        return points
    return np.unique(points.astype(np.int32), axis=0)


def build_points(img, mask, detail, transition_sens):
    """
    detail: 0..100
    transition_sens: 0..100, higher = more internal boundaries
    """
    t = detail / 100.0
    s = transition_sens / 100.0

    contour_step = lerp(12, 3, t)
    edge_step = lerp(12, 3, t)
    grid_step = lerp(24, 8, t)
    random_points = lerp(120, 1200, t)

    canny1 = lerp(140, 35, s)
    canny2 = lerp(250, 110, s)
    min_score = lerp(75, 12, s)

    pts = [
        border_points(img, step=max(20, grid_step * 2)),
        largest_contour_points(mask, step=contour_step),
        edge_points(img, mask=mask, canny1=canny1, canny2=canny2, step=edge_step),
        transition_points(img, mask=mask, block=grid_step, min_score=min_score),
        random_interior_points(mask, count=random_points),
    ]

    pts = np.vstack([p for p in pts if len(p) > 0])
    pts = unique_points(pts)
    return pts


@njit(cache=True, fastmath=True)
def triangle_area(pts):
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    return abs(
        x1 * (y2 - y3) +
        x2 * (y3 - y1) +
        x3 * (y1 - y2)
    ) / 2.0


def triangle_color_mean(img, tri_pts, mask=None):
    tri_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(tri_mask, tri_pts.astype(np.int32), 255)

    if mask is not None:
        tri_mask = cv2.bitwise_and(tri_mask, tri_mask, mask=mask)

    if cv2.countNonZero(tri_mask) == 0:
        cx = int(np.clip(np.mean(tri_pts[:, 0]), 0, img.shape[1] - 1))
        cy = int(np.clip(np.mean(tri_pts[:, 1]), 0, img.shape[0] - 1))
        b, g, r = img[cy, cx]
        return int(b), int(g), int(r)

    mean = cv2.mean(img, mask=tri_mask)
    return int(mean[0]), int(mean[1]), int(mean[2])


def gradient_strength_map(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    score = np.zeros(img.shape[:2], dtype=np.float32)

    for ch in cv2.split(lab):
        sx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        score += cv2.magnitude(sx, sy)

    score = cv2.GaussianBlur(score, (5, 5), 0)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX)
    return score.astype(np.float32)


# @njit(cache=True, fastmath=True)
def sample_line_strength(p1, p2, grad_map, samples=16):
    xs = np.linspace(p1[0], p2[0], samples)
    ys = np.linspace(p1[1], p2[1], samples)

    h, w = grad_map.shape
    vals = []

    for x, y in zip(xs, ys):
        ix = int(np.clip(round(x), 0, w - 1))
        iy = int(np.clip(round(y), 0, h - 1))
        vals.append(grad_map[iy, ix])

    return float(np.mean(vals)) if vals else 0.0


def draw_triangle_edges(out, tri_pts, grad_map, line_max, edge_sens):
    """
    Lines at strong transitions are thicker, while lines that simply close the triangle shape remain thin.
    """
    strong_thr = lerp(85, 25, edge_sens / 100.0)

    edges = [
        (tri_pts[0], tri_pts[1]),
        (tri_pts[1], tri_pts[2]),
        (tri_pts[2], tri_pts[0]),
    ]

    for a, b in edges:
        strength = sample_line_strength(a, b, grad_map, samples=18)

        if strength >= strong_thr:
            t = (strength - strong_thr) / max(1.0, 255.0 - strong_thr)
            thickness = 1 + int(t * max(1, line_max - 1))
        else:
            thickness = 1

        cv2.line(
            out,
            tuple(a.astype(int)),
            tuple(b.astype(int)),
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )


def triangulate_cubist(img, use_fg=True, fg_margin=0.06, detail=50, transition_sens=55, line_max=4):
    mask = foreground_mask(img, use_fg=use_fg, margin=fg_margin)
    points = build_points(img, mask, detail=detail, transition_sens=transition_sens)

    if len(points) < 3:
        raise RuntimeError("Too few points for triangulation.")

    tri = Delaunay(points)
    out = np.full_like(img, 255)
    grad_map = gradient_strength_map(img)

    h, w = img.shape[:2]

    for simplex in tri.simplices:
        tri_pts = points[simplex].astype(np.int32)

        if triangle_area(tri_pts) < 8:
            continue

        tri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, tri_pts, 255)

        inside = cv2.bitwise_and(tri_mask, tri_mask, mask=mask)
        coverage = cv2.countNonZero(inside) / max(1, cv2.countNonZero(tri_mask))

        # To avoid holes along the edge of the silhouette:
        # We take triangles that at least partially belong to the object.
        if coverage < 0.12:
            continue

        color = triangle_color_mean(img, tri_pts, mask=mask)
        cv2.fillConvexPoly(out, tri_pts, color)

    # We draw lines over the already filled picture:
    for simplex in tri.simplices:
        tri_pts = points[simplex].astype(np.int32)

        if triangle_area(tri_pts) < 8:
            continue

        tri_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, tri_pts, 255)
        inside = cv2.bitwise_and(tri_mask, tri_mask, mask=mask)
        coverage = cv2.countNonZero(inside) / max(1, cv2.countNonZero(tri_mask))

        if coverage < 0.12:
            continue

        draw_triangle_edges(out, tri_pts, grad_map, line_max=line_max, edge_sens=transition_sens)

    return out, mask


def nothing(x):
    pass


def main():
    parser = argparse.ArgumentParser(description="Cubist triangulation with live sliders.")
    parser.add_argument("input", help="Path to image")
    parser.add_argument("-o", "--output", help="Where to save the result")
    args = parser.parse_args()

    img = load_image(args.input)

    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("Detail", "controls", 45, 100, nothing)
    cv2.createTrackbar("Transition sens", "controls", 60, 100, nothing)
    cv2.createTrackbar("Line max", "controls", 4, 12, nothing)
    cv2.createTrackbar("Use FG", "controls", 1, 1, nothing)
    cv2.createTrackbar("FG margin", "controls", 6, 20, nothing)

    last_params = None
    last_result = None
    last_mask = None

    while True:
        detail = cv2.getTrackbarPos("Detail", "controls")
        transition_sens = cv2.getTrackbarPos("Transition sens", "controls")
        line_max = max(1, cv2.getTrackbarPos("Line max", "controls"))
        use_fg = cv2.getTrackbarPos("Use FG", "controls") == 1
        fg_margin = cv2.getTrackbarPos("FG margin", "controls") / 100.0

        params = (detail, transition_sens, line_max, use_fg, fg_margin)

        if params != last_params:
            try:
                result, mask = triangulate_cubist(
                    img,
                    use_fg=use_fg,
                    fg_margin=fg_margin,
                    detail=detail,
                    transition_sens=transition_sens,
                    line_max=line_max
                )
                last_result = result
                last_mask = mask
                last_params = params

                cv2.imshow("preview", last_result)
                cv2.imshow("mask", last_mask * 255)
            except Exception as e:
                err = np.full_like(img, 255)
                cv2.putText(err, str(e), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("preview", err)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s") and last_result is not None:
            cv2.imwrite(args.output, last_result)
            print(f"Save: {args.output}")

    if last_result is not None:
        cv2.imwrite(args.output, last_result)
        print(f"Save: {args.output}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    np.random.seed(42)
    main()
