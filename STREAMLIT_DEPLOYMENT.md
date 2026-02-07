# Streamlit Deployment Guide

## Overview

This guide covers local development and production deployment of the Streamlit dashboard for the Misinformation at Scale project.

---

## 🚀 Local Development

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Installation & Setup

```bash
# 1. Navigate to project directory
cd c:\Users\sanja\OneDrive\Documents\GitHub\misinformation-at-scale

# 2. Create virtual environment (recommended)
python -m venv streamlit_env
streamlit_env\Scripts\activate  # Windows
# OR source streamlit_env/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r dashboard_requirements.txt

# 4. Run locally
streamlit run app.py
```

**Access**: http://localhost:8501

### Development with Auto-reload
```bash
streamlit run app.py --logger.level=debug
```

### Test with Different Port
```bash
streamlit run app.py --server.port=8502
```

---

## 🌐 Cloud Deployment Options

### Option A: Streamlit Cloud (Recommended - Easiest)

**Cost**: Free tier available  
**Setup Time**: 5 minutes  
**Best For**: Quick deployment, no infrastructure

#### Steps:

1. **Push to GitHub** (already done - `main` branch)
   ```bash
   git status
   git push
   ```

2. **Connect to Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select: 
     - GitHub repo: `sanjaykshetri/misinformation-at-scale`
     - Branch: `main`
     - File path: `app.py`
   - Click Deploy

3. **Configure secrets** (if needed):
   - Go to app settings → Secrets
   - Add API keys or credentials
   - Stored securely, never committed to git

**Result**: Live URL like `https://misinformation-at-scale.streamlit.app`

---

### Option B: Docker Container (Most Reliable)

**Cost**: Free (host anywhere: AWS, Google Cloud, DigitalOcean, etc.)  
**Setup Time**: 15-20 minutes  
**Best For**: Production, self-hosted, reproducibility

#### Local Testing with Docker:

```bash
# 1. Install Docker: https://www.docker.com/products/docker-desktop

# 2. Build image
docker build -t misinformation-dashboard .

# 3. Run container
docker run -p 8501:8501 misinformation-dashboard

# 4. Access at http://localhost:8501
```

#### Deploy to Cloud Service:

**AWS EC2:**
```bash
# Push to ECR, then run on EC2
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag misinformation-dashboard:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/misinformation-dashboard:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/misinformation-dashboard:latest
```

**Google Cloud Run:**
```bash
# Deploy serverless
gcloud run deploy misinformation-dashboard --source . --platform managed
```

**DigitalOcean App Platform:**
- Connect GitHub repo directly
- Auto-deploys on push to main
- Free tier available

---

### Option C: Traditional Server

**Cost**: ~$5-10/month (DigitalOcean, Linode, etc.)  
**Setup Time**: 30 minutes  
**Best For**: Full control, always-on deployment

```bash
# 1. SSH into server
ssh root@your_server_ip

# 2. Install Python
sudo apt update && sudo apt install python3.11 python3-pip

# 3. Clone repo
git clone https://github.com/sanjaykshetri/misinformation-at-scale.git
cd misinformation-at-scale

# 4. Install dependencies
pip install -r dashboard_requirements.txt

# 5. Run with systemd (persistent)
sudo nano /etc/systemd/system/streamlit.service
```

Add this to the service file:
```ini
[Unit]
Description=Streamlit Misinformation Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/misinformation-at-scale
ExecStart=/usr/bin/python3 -m streamlit run app.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit
```

---

## ✅ Deployment Checklist

- [ ] Clean up development files (`.ipynb_checkpoints`, `__pycache__`)
- [ ] Update `dashboard_requirements.txt` with pinned versions:
  ```bash
  pip freeze > dashboard_requirements.txt
  ```
- [ ] Test locally: `streamlit run app.py`
- [ ] Verify no hardcoded paths (use relative paths)
- [ ] Add `.streamlit/secrets.toml` if using API keys (in `.gitignore`)
- [ ] Test with `--headless` mode: `streamlit run app.py --headless`
- [ ] Add health check endpoint (for production monitoring)
- [ ] Configure error logging
- [ ] Test on target deployment platform

---

## 🔧 Troubleshooting

### Issue: "Module not found" on deployment
**Solution**: Ensure all imports in `app.py` are in `dashboard_requirements.txt`
```bash
pip show streamlit pandas numpy matplotlib seaborn scikit-learn
```

### Issue: Port already in use
**Solution**: Use different port or kill process
```bash
# Linux/macOS
lsof -i :8501
kill -9 <PID>

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue: Slow cold start on cloud
**Solution**: Use slim Docker image, pre-cache dependencies
```dockerfile
RUN pip install --no-cache-dir -r dashboard_requirements.txt
```

### Issue: Memory errors on free tier
**Solution**: 
```python
# In app.py, add at top
import streamlit as st
st.set_page_config(
    memory_limit=512,  # MB
    page_title="Dashboard"
)
```

---

## 📊 Monitoring & Maintenance

### Streamlit Cloud
- Auto-updates on git push
- View logs in dashboard
- Set up GitHub Actions for testing

### Docker/Self-hosted
```bash
# Monitor logs
docker logs -f misinformation-dashboard

# Auto-restart on failure
docker run --restart=always -p 8501:8501 misinformation-dashboard
```

### Health Check API
```python
# Add to app.py
@st.api.cache_resource
def health_check():
    return {"status": "healthy", "version": "1.0"}
```

---

## 🚀 Production Best Practices

1. **Use secrets management**:
   - `.streamlit/secrets.toml` (never commit)
   - Cloud provider secret services
   - Environment variables

2. **Enable HTTPS**:
   - Use reverse proxy (nginx)
   - Let's Encrypt certificates
   - Cloudflare for DNS

3. **Monitor performance**:
   - Add logging
   - Track user interactions
   - Monitor server resources

4. **Version control**:
   - Tag releases: `git tag v1.0.0`
   - Semantic versioning
   - Changelog documentation

5. **Testing before deploy**:
   ```bash
   pytest tests/
   streamlit run app.py --headless --logger.level=debug
   ```

---

## 🎯 Recommended Deployment Path

For this project:

1. **Immediate**: Streamlit Cloud (free, easy)
   ```
   https://share.streamlit.io → GitHub → Auto-deploy
   ```

2. **Production**: Docker on DigitalOcean App Platform
   ```
   DigitalOcean → GitHub integration → Auto-deploy from main
   ```

3. **Advanced**: Kubernetes cluster with auto-scaling
   ```
   GKE/EKS → Docker → Helm charts → CI/CD pipeline
   ```

---

## 📚 Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Docker Docs](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)

---

**Status**: Ready for deployment  
**Last Updated**: February 2026  
**Maintainer**: Sanjay K. Shetri
