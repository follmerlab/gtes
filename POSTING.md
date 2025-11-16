# Quick Post Guide

## Create a New Post

### Ground State (Fundamentals)
```bash
hugo new content/ground-state/my-post-title.md
```

### Excited State (Hot Takes)
```bash
hugo new content/excited-state/my-post-title.md
```

## Edit Your Post

1. Open the file in your editor
2. Update the front matter:
   - Change `draft = false` when ready to publish
   - Add relevant tags
   - Write a compelling summary
3. Write your content with Markdown + LaTeX

## Publish

```bash
git add .
git commit -m "New post: Your Title"
git push origin main
```

## LaTeX Examples

Inline: `$E = mc^2$`

Display:
```
$$
\hat{H}\psi = E\psi
$$
```

## Preview Locally

```bash
hugo server -D
```

Visit: http://localhost:1313
