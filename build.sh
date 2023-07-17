#docker buildx build --platform linux/arm64 --target arm64_stage --push -t lorenzoinm/teaching-ai-toolkit:arm64 .
docker buildx build --platform linux/arm64 --push -t lorenzoinm/teaching-ai-toolkit:arm64 -f Dockerfile_reduced .
