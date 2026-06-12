# CadQuery

Python-based parametric CAD library built on the OpenCascade kernel. CadQuery
models are B-reps and should be exported as STEP for this benchmark.

**Default units: millimeters (mm).**

## How your code is run

- Write a single self-contained ```python block.
- Build a `cadquery.Workplane` / `Shape` / `Assembly` and export `output.step`.
- Use `print(...)` for diagnostics.
- The harness validates `output.step` and sends back a render.

## Core pattern

```python
import cadquery as cq
from cadquery import exporters

part = (
    cq.Workplane("XY")
    .box(60, 40, 8)
    .faces(">Z").workplane()
    .hole(8)
)

exporters.export(part, "output.step")
print("exported output.step")
```

## Workplanes and selectors

```python
cq.Workplane("XY")             # sketch in XY, normal +Z
.faces(">Z")                   # top face
.faces("<Z")                   # bottom face
.faces("|Z")                   # vertical faces
.edges("|Z")                   # vertical edges
.vertices()                    # selected vertices
.workplane(centerOption="CenterOfMass")
```

Useful selector strings: `>X`, `<X`, `>Y`, `<Y`, `>Z`, `<Z`, `|Z`, `%Circle`,
`not(...)`, `and`, `or`.

## Solids and sketches

```python
cq.Workplane("XY").box(x, y, z)
cq.Workplane("XY").circle(r).extrude(h)
cq.Workplane("XY").rect(w, h).extrude(h)
cq.Workplane("XY").polygon(n, diameter).extrude(h)
cq.Workplane("XY").polyline([(x1,y1), ...]).close().extrude(h)
cq.Workplane("XY").sphere(r)
cq.Workplane("XY").cylinder(h, r)
```

## Holes, pockets, slots

```python
part.faces(">Z").workplane().hole(diameter)
part.faces(">Z").workplane().cboreHole(d, cboreDiameter, cboreDepth)
part.faces(">Z").workplane().cskHole(d, cskDiameter, cskAngle)
part.faces(">Z").workplane().slot2D(length, diameter).cutThruAll()
part.faces(">Z").workplane().rect(w, h).cutBlind(depth)
```

For through cuts, prefer `cutThruAll()` or make cutting solids extend past both
faces to avoid coincident residual faces.

## Fillets and chamfers

CadQuery has real BREP fillets/chamfers. Use them for rounded machined parts:

```python
part = part.edges("|Z").fillet(2)
part = part.edges(">Z").chamfer(1)
```

Select narrowly. If a fillet fails, reduce the radius or select fewer edges.

## Booleans and transforms

```python
part = base.union(boss)
part = base.cut(tool)
part = a.intersect(b)

part = part.translate((x, y, z))
part = part.rotate((0,0,0), (0,0,1), 45)   # axis-angle, degrees
part = part.mirror("YZ")
```

## Repetition

```python
for x, y in [(20, 0), (-20, 0)]:
    part = part.union(cq.Workplane("XY").center(x, y).circle(5).extrude(10))

part = part.faces(">Z").workplane().pushPoints([(20,0), (-20,0)]).hole(6)
```

## Revolved / turned profiles

```python
profile = (
    cq.Workplane("XZ")
    .polyline([(0,0), (20,0), (20,10), (12,14), (0,14)])
    .close()
)
part = profile.revolve(360, (0,0,0), (0,0,1))
```

## Assemblies

Use assemblies only when helpful; a single compound/part is usually simpler.

```python
assy = cq.Assembly()
assy.add(part1)
assy.add(part2, loc=cq.Location(cq.Vector(20, 0, 0)))
exporters.export(assy, "output.step")
```

## Editing a STEP

```python
import cadquery as cq
from cadquery import importers, exporters

shape = importers.importStep("input.step")
# modify / combine shape...
exporters.export(shape, "output.step")
```

## Good benchmark habits

- Define dimensions as variables at the top.
- Keep the bbox centered near the origin and orient the longest axis along X.
- Prefer real fillets/chamfers over faceted approximations.
- Export only `output.step`; do not rely on viewer-only objects.
