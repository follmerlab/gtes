# Content Organization

This site uses Hugo's **page bundles** structure for organizing posts and their associated resources.

## Structure

Each post should be in its own folder with:
- `index.md` - The main markdown content
- Images, PDFs, data files, or other resources in the same folder

```
content/
├── ground-state/
│   ├── _index.md                                    # Section landing page
│   ├── understanding-linear-regression/
│   │   ├── index.md                                 # Post content
│   │   ├── linear-regression-analysis.png           # Post image
│   │   └── [other resources]                        # Data, notebooks, etc.
│   └── kbeta-explainer/
│       └── index.md
├── excited-state/
│   ├── _index.md
│   └── ultrafast-heme-notes/
│       └── index.md
└── posts/
    └── _index.md
```

## Benefits

1. **Organization**: All resources for a post stay together
2. **Portability**: Easy to move or archive entire posts
3. **Relative links**: Reference images and files with simple paths like `image.png`
4. **Version control**: Clear history for each post and its assets

## Adding a New Post

1. Create a new folder in `ground-state/` or `excited-state/`:
   ```bash
   mkdir -p content/ground-state/my-new-post
   ```

2. Create `index.md` with frontmatter:
   ```markdown
   +++
   title = "My Post Title"
   date = 2025-11-19T10:00:00-08:00
   draft = false
   tags = ["tag1", "tag2"]
   categories = ["category"]
   summary = "Brief description"
   +++
   
   # Content here...
   ```

3. Add images, data files, etc. to the same folder

4. Reference resources with relative paths:
   ```markdown
   ![Description](my-image.png)
   [Download data](data.csv)
   ```

## Hugo Page Bundles

Hugo recognizes folders with `index.md` as **leaf bundles**. Resources in the same folder are automatically associated with the page and served at the same URL path.

For example:
- Content at: `content/ground-state/my-post/index.md`
- Image at: `content/ground-state/my-post/figure.png`
- Results in URLs:
  - Post: `https://follmerlab.github.io/gtes/ground-state/my-post/`
  - Image: `https://follmerlab.github.io/gtes/ground-state/my-post/figure.png`
