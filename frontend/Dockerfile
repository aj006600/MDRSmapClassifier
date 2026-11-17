# --platform=$BUILDPLATFORM:npm ci 與 vite build 的產物是與架構無關的靜態檔,
# 所以這個 stage 一律在 build 主機的原生架構上跑,不必模擬目標架構
FROM --platform=$BUILDPLATFORM node:22-slim AS build
# 專案 label:make clean 只清帶它的 image;兩個 stage 都要帶,build stage 留下的舊 image 才清得到
LABEL org.opencontainers.image.title="mdrsmap-classifier"
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
LABEL org.opencontainers.image.title="mdrsmap-classifier"
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
