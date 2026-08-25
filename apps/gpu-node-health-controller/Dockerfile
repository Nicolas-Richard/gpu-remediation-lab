FROM golang:1.22-bookworm AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go test ./...
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /gpu-node-health-controller ./cmd/controller

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /gpu-node-health-controller /gpu-node-health-controller

USER 65532:65532
ENTRYPOINT ["/gpu-node-health-controller"]
