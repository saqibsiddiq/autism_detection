FROM node:20-bookworm

# Create app directory
WORKDIR /app

# Install dependencies first (leverage Docker layer caching)
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund

# Copy source
COPY server.js ./

# Expose port
EXPOSE 5001

# Run server
CMD ["npm", "start"]


