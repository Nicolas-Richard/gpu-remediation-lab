package main

import (
	"net/http"
	"testing"

	healthsource "hackweek/gpu-node-health-controller/internal/health/source"
)

func TestNewHealthSourceSelectsExactlyOneConfiguredSource(t *testing.T) {
	httpClient := &http.Client{}
	config := healthSourceConfig{
		dcgmExporterNamespace:  "gpu-operator",
		dcgmExporterSelector:   " app=nvidia-dcgm-exporter ",
		dcgmSimulatorNamespace: "gpu-node-health-system",
		httpClient:             httpClient,
	}

	t.Run("annotation", func(t *testing.T) {
		source, err := newHealthSource(healthSourceAnnotation, config)
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := source.(*healthsource.Annotation); !ok {
			t.Fatalf("source type = %T, want *source.Annotation", source)
		}
	})

	t.Run("dcgm", func(t *testing.T) {
		source, err := newHealthSource(healthSourceDCGM, config)
		if err != nil {
			t.Fatal(err)
		}
		dcgm, ok := source.(*healthsource.DCGM)
		if !ok {
			t.Fatalf("source type = %T, want *source.DCGM", source)
		}
		if dcgm.Namespace != config.dcgmExporterNamespace || dcgm.SourceName != "" {
			t.Fatalf("DCGM source = %#v, want real exporter configuration", dcgm)
		}
		if dcgm.LabelSelector != "app=nvidia-dcgm-exporter" {
			t.Fatalf("DCGM selector = %q, want trimmed configured selector", dcgm.LabelSelector)
		}
		if dcgm.HTTPClient != httpClient {
			t.Fatal("DCGM source did not retain the configured HTTP client")
		}
	})

	t.Run("dcgm simulator", func(t *testing.T) {
		source, err := newHealthSource("  "+healthSourceDCGMSimulator+"  ", config)
		if err != nil {
			t.Fatal(err)
		}
		dcgm, ok := source.(*healthsource.DCGM)
		if !ok {
			t.Fatalf("source type = %T, want *source.DCGM", source)
		}
		if dcgm.Namespace != config.dcgmSimulatorNamespace || dcgm.SourceName != healthSourceDCGMSimulator {
			t.Fatalf("DCGM source = %#v, want simulator configuration", dcgm)
		}
		if got := dcgm.Labels["gpu-orch.dev/health-exporter"]; got != healthSourceDCGMSimulator {
			t.Fatalf("simulator selector = %q, want %q", got, healthSourceDCGMSimulator)
		}
	})
}

func TestNewHealthSourceRejectsMissingOrUnsupportedSource(t *testing.T) {
	for _, name := range []string{"", "auto", "annotation,dcgm"} {
		t.Run(name, func(t *testing.T) {
			if _, err := newHealthSource(name, healthSourceConfig{}); err == nil {
				t.Fatalf("newHealthSource(%q) returned no error", name)
			}
		})
	}
}

func TestNewHealthSourceRejectsInvalidDCGMExporterSelector(t *testing.T) {
	for _, selector := range []string{"", "app in ("} {
		t.Run(selector, func(t *testing.T) {
			_, err := newHealthSource(healthSourceDCGM, healthSourceConfig{
				dcgmExporterSelector: selector,
			})
			if err == nil {
				t.Fatalf("newHealthSource(dcgm, selector=%q) returned no error", selector)
			}
		})
	}
}
