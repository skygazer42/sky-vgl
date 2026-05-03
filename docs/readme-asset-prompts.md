# README Asset Prompt

Use this single prompt as the style and composition baseline for generating README visuals for VGL. Replace the bracketed asset description with the specific image you want, for example `a logo banner`, `a graph types overview`, `an architecture diagram`, `a training pipeline`, or `an operator coverage board`.

```text
Create [ASSET DESCRIPTION] for an open-source Python graph learning package called VGL, short for Versatile Graph Learning.

The image should feel like a polished Python developer tool asset, not a marketing landing page and not a corporate ad. Aim for the visual language of a modern open-source Python package README: clean white background, precise geometry, crisp edges, restrained gradients, strong spacing, technically credible, calm, minimal, and documentation-friendly.

The design should communicate that VGL is a serious graph learning toolkit for Python with one unified surface across homogeneous graphs, heterogeneous graphs, temporal graphs, training, sampling, and model building. Favor modular structure, developer clarity, and a package-architecture mindset over visual spectacle.

Use a premium technical illustration style, vector-inspired rather than photorealistic. Keep the palette in the Python tooling family: indigo, cobalt blue, violet, cyan, emerald accents, soft slate neutrals, and lots of white space. The result should look excellent embedded in a GitHub README.

Avoid clutter, dense paragraphs of text, browser chrome, device mockups, fake dashboards, photoreal scenes, people, noisy textures, stock-art energy, heavy shadows, watermark, and anything that feels like a startup ad. If text appears at all, keep it minimal, sharp, and secondary to the graphic structure.
```

Recommended generation settings:

- drafts: `1024x1024` or `1536x1024`, `quality=low`
- final wide README assets: `2048x1152` or `2048x768`, `quality=high`
