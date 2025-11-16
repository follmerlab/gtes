# Ground Truths & Excited States

A blog exploring the intersection of fundamental science and cutting-edge research in molecular chemistry and spectroscopy.

**Live Site:** (will be added after GitHub Pages deployment)

## 🚀 Quick Start: Writing & Publishing Posts

### Creating a New Post

#### For Ground State Posts (Fundamentals & Philosophy)

```bash
cd /Users/afollmer/GTES/gt-es-website
hugo new content/ground-state/your-post-title.md
```

#### For Excited State Posts (Hot Takes & New Results)

```bash
cd /Users/afollmer/GTES/gt-es-website
hugo new content/excited-state/your-post-title.md
```

This creates a new post file with the proper template, including:
- Front matter (title, date, tags, categories)
- LaTeX math support examples
- Markdown structure

### Writing Your Post

1. **Open the created file** in your editor
2. **Edit the front matter:**
   ```toml
   +++
   title = "Your Awesome Title"
   date = 2025-11-15T10:00:00-08:00
   draft = false  # Change to false when ready to publish
   tags = ["xes", "spin-state", "metalloproteins"]
   categories = ["spectroscopy"]
   summary = "A compelling one-sentence summary"
   +++
   ```
3. **Write your content** using Markdown and LaTeX

### LaTeX Math Support

**Inline math:** `$E = mc^2$` renders as $E = mc^2$

**Display equations:**
```latex
$$
\hat{H}\psi = E\psi
$$
```

**Complex equations:**
```latex
$$
\begin{aligned}
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{B} &= \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}
\end{aligned}
$$
```

### Publishing Your Post

Once you're happy with your post:

```bash
# 1. Save your changes
# 2. Set draft = false in the front matter
# 3. Commit and push to GitHub

git add .
git commit -m "New post: Your Post Title"
git push origin main
```

**That's it!** GitHub Actions will automatically:
- Build your Hugo site
- Deploy to GitHub Pages
- Your post goes live in ~1 minute

## 📝 Post Structure

### Ground State Posts
- Fundamentals in science
- Philosophical perspectives on modeling
- Careful analysis of core principles
- Deep dives into established methods

### Excited State Posts
- Hot takes on new results
- Cutting-edge experimental findings
- Incomplete thoughts and speculations
- Time-resolved ideas

## 🛠️ Local Development

### Preview Your Site Locally

```bash
cd /Users/afollmer/GTES/gt-es-website
hugo server -D
```

Visit `http://localhost:1313` to see your site with drafts included.

### Build the Site

```bash
hugo
```

This generates the static site in the `/public` directory.

## 📁 Repository Structure

```
gt-es-website/
├── archetypes/          # Post templates
│   ├── ground-state.md
│   └── excited-state.md
├── content/             # Your posts
│   ├── ground-state/
│   ├── excited-state/
│   └── about.md
├── layouts/             # HTML templates
├── static/              # CSS, images, etc.
│   └── css/styles.css
├── config.toml          # Site configuration
└── README.md
```

## 🎨 Customization

### Adding Images

1. Place images in `/static/images/`
2. Reference in posts: `![Alt text](/images/figure.png)`

### Editing the Theme

- **Colors & Styling:** Edit `/static/css/styles.css`
- **Layout:** Edit files in `/layouts/`
- **Site Config:** Edit `config.toml`

## 🔧 Hugo Commands Reference

| Command | Description |
|---------|-------------|
| `hugo new content/ground-state/post.md` | Create new Ground State post |
| `hugo new content/excited-state/post.md` | Create new Excited State post |
| `hugo server -D` | Run local dev server (includes drafts) |
| `hugo server` | Run local dev server (published only) |
| `hugo` | Build static site |

## 📚 Useful Tags

Common tags for organizing posts:
- `xes`, `xas`, `xfel` - X-ray techniques
- `tr-sfx`, `time-resolved` - Time-resolved methods
- `spin-state`, `kβ`, `emission` - Spectroscopic properties
- `metalloproteins`, `p450` - Biological systems
- `dft`, `tddft` - Computational methods

## 🚀 Deployment

The site automatically deploys via GitHub Actions when you push to `main`. No manual build steps needed!

## 📖 Writing Tips

1. **Set draft = false** when ready to publish
2. **Use descriptive titles** that capture the main idea
3. **Add tags** to help readers find related content
4. **Include a summary** for post listings
5. **Use LaTeX** for equations and mathematical notation
6. **Break up text** with headers and lists for readability

## 🆘 Troubleshooting

**Post not showing up?**
- Check that `draft = false` in the front matter
- Ensure the date isn't in the future

**Math not rendering?**
- Use `$...$` for inline math
- Use `$$...$$` for display equations
- Ensure proper escaping in complex equations

**Local preview not working?**
- Make sure you're in the `/gt-es-website` directory
- Try stopping and restarting: `pkill hugo && hugo server -D`

---

**Questions?** Open an issue or check the [Hugo documentation](https://gohugo.io/documentation/).
