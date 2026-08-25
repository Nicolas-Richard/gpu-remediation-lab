package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"hackweek/gpu-node-health-controller/internal/controller"
	healthsource "hackweek/gpu-node-health-controller/internal/health/source"
)

const (
	healthSourceAnnotation    = "annotation"
	healthSourceDCGM          = "dcgm"
	healthSourceDCGMSimulator = "dcgm-simulator"
)

type healthSourceConfig struct {
	podReader              client.Reader
	dcgmExporterNamespace  string
	dcgmExporterSelector   string
	dcgmSimulatorNamespace string
	httpClient             *http.Client
}

func main() {
	var metricsAddress string
	var healthAddress string
	var leaderElection bool
	var healthPollInterval time.Duration
	var recoveryConfirmationWindow time.Duration
	var healthSourceName string
	var dcgmExporterNamespace string
	var dcgmExporterSelector string
	var dcgmSimulatorNamespace string

	flag.StringVar(&metricsAddress, "metrics-bind-address", "0", "Metrics endpoint address; use 0 to disable")
	flag.StringVar(&healthAddress, "health-probe-bind-address", ":8081", "Health probe endpoint address")
	flag.BoolVar(&leaderElection, "leader-elect", false, "Enable leader election")
	flag.DurationVar(&healthPollInterval, "health-poll-interval", 10*time.Second, "Interval between node health observations")
	flag.DurationVar(
		&recoveryConfirmationWindow,
		"recovery-confirmation-window",
		20*time.Second,
		"Continuous healthy observation window required before clearing isolation",
	)
	flag.StringVar(
		&healthSourceName,
		"health-source",
		"",
		"Required health source: annotation, dcgm, or dcgm-simulator",
	)
	flag.StringVar(
		&dcgmExporterNamespace,
		"dcgm-exporter-namespace",
		"gpu-operator",
		"Namespace containing per-node DCGM exporter pods",
	)
	flag.StringVar(
		&dcgmExporterSelector,
		"dcgm-exporter-label-selector",
		"app=nvidia-dcgm-exporter,component=status-exporter",
		"Kubernetes label selector identifying per-node DCGM exporter pods",
	)
	flag.StringVar(
		&dcgmSimulatorNamespace,
		"dcgm-simulator-namespace",
		"gpu-node-health-system",
		"Namespace containing the local per-node DCGM metrics simulator",
	)

	loggingOptions := zap.Options{Development: true}
	loggingOptions.BindFlags(flag.CommandLine)
	flag.Parse()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&loggingOptions)))

	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		ctrl.Log.Error(err, "unable to add Kubernetes core types to scheme")
		os.Exit(1)
	}

	config := ctrl.GetConfigOrDie()
	manager, err := ctrl.NewManager(config, ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsserver.Options{BindAddress: metricsAddress},
		HealthProbeBindAddress: healthAddress,
		LeaderElection:         leaderElection,
		LeaderElectionID:       "gpu-node-health-controller.gpu-orch.dev",
	})
	if err != nil {
		ctrl.Log.Error(err, "unable to create manager")
		os.Exit(1)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		ctrl.Log.Error(err, "unable to create Kubernetes clientset")
		os.Exit(1)
	}
	configuredHealthSource, err := newHealthSource(healthSourceName, healthSourceConfig{
		podReader:              manager.GetAPIReader(),
		dcgmExporterNamespace:  dcgmExporterNamespace,
		dcgmExporterSelector:   dcgmExporterSelector,
		dcgmSimulatorNamespace: dcgmSimulatorNamespace,
		httpClient:             &http.Client{Timeout: 5 * time.Second},
	})
	if err != nil {
		ctrl.Log.Error(err, "unable to configure health source")
		os.Exit(1)
	}

	reconciler := &controller.NodeHealthReconciler{
		Client:         manager.GetClient(),
		PodReader:      manager.GetAPIReader(),
		Evicter:        controller.NewKubernetesPodEvicter(clientset),
		Recorder:       manager.GetEventRecorderFor("gpu-node-health-controller"),
		HealthSource:   configuredHealthSource,
		PollInterval:   healthPollInterval,
		RecoveryWindow: recoveryConfirmationWindow,
	}
	if err := reconciler.SetupWithManager(manager); err != nil {
		ctrl.Log.Error(err, "unable to register node controller")
		os.Exit(1)
	}

	if err := manager.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		ctrl.Log.Error(err, "unable to configure health check")
		os.Exit(1)
	}
	if err := manager.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		ctrl.Log.Error(err, "unable to configure readiness check")
		os.Exit(1)
	}

	ctrl.Log.Info(
		"starting GPU node health controller",
		"healthSource",
		strings.TrimSpace(healthSourceName),
	)
	if err := manager.Start(ctrl.SetupSignalHandler()); err != nil {
		ctrl.Log.Error(err, "manager stopped with an error")
		os.Exit(1)
	}
}

func newHealthSource(name string, config healthSourceConfig) (healthsource.Source, error) {
	switch strings.TrimSpace(name) {
	case healthSourceAnnotation:
		return &healthsource.Annotation{}, nil
	case healthSourceDCGM:
		selector := strings.TrimSpace(config.dcgmExporterSelector)
		if selector == "" {
			return nil, fmt.Errorf("DCGM exporter label selector must not be empty")
		}
		if _, err := labels.Parse(selector); err != nil {
			return nil, fmt.Errorf("parse DCGM exporter label selector %q: %w", selector, err)
		}
		return &healthsource.DCGM{
			PodReader:     config.podReader,
			Namespace:     config.dcgmExporterNamespace,
			LabelSelector: selector,
			HTTPClient:    config.httpClient,
		}, nil
	case healthSourceDCGMSimulator:
		return &healthsource.DCGM{
			PodReader:  config.podReader,
			Namespace:  config.dcgmSimulatorNamespace,
			SourceName: healthSourceDCGMSimulator,
			Labels: map[string]string{
				"gpu-orch.dev/health-exporter": healthSourceDCGMSimulator,
			},
			HTTPClient: config.httpClient,
		}, nil
	default:
		return nil, fmt.Errorf(
			"unsupported health source %q; expected %s, %s, or %s",
			name,
			healthSourceAnnotation,
			healthSourceDCGM,
			healthSourceDCGMSimulator,
		)
	}
}
