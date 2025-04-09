let comparisonChart;

document.getElementById("predictForm").addEventListener("submit", function(e) {
    e.preventDefault();
    const formData = {
        vendor_id: this.vendor_id.value,
        payment_amount: this.payment_amount.value,
        vendor_category: this.vendor_category.value,
        vendor_location: this.vendor_location.value,
        hist_payment_behavior: this.hist_payment_behavior.value,
        invoice_date: this.invoice_date.value,
        due_date: this.due_date.value
    };

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML = `Predicted Delay: ${data.delay_days.toFixed(2)} days<br>Anomaly: ${data.is_anomaly}`;

        const ctx = document.getElementById("comparisonChart").getContext("2d");
        if (comparisonChart) {
            comparisonChart.destroy();
        }
        comparisonChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Predicted Delay", "Dataset Mean Delay"],
                datasets: [{
                    label: "Delay (days)",
                    data: [data.comparison.predicted_delay, data.comparison.mean_delay],
                    backgroundColor: ["rgba(255, 99, 132, 0.2)", "rgba(54, 162, 235, 0.2)"],
                    borderColor: ["rgba(255, 99, 132, 1)", "rgba(54, 162, 235, 1)"],
                    borderWidth: 1
                }]
            },
            options: {
                scales: { y: { beginAtZero: true } },
                plugins: { title: { display: true, text: "Comparison of Predicted Delay vs Dataset Mean" } }
            }
        });
    });
});

// Load visualization data
fetch("/data")
    .then(response => response.json())
    .then(data => {
        const delayCtx = document.getElementById("delayChart").getContext("2d");
        new Chart(delayCtx, {
            type: "bar",
            data: {
                labels: Object.keys(data.delay_dist),
                datasets: [{
                    label: "Delay Distribution",
                    data: Object.values(data.delay_dist),
                    backgroundColor: "rgba(75, 192, 192, 0.2)",
                    borderColor: "rgba(75, 192, 192, 1)",
                    borderWidth: 1
                }]
            },
            options: { scales: { y: { beginAtZero: true } } }
        });

        const anomalyCtx = document.getElementById("anomalyChart").getContext("2d");
        new Chart(anomalyCtx, {
            type: "pie",
            data: {
                labels: ["Normal", "Anomaly"],
                datasets: [{
                    data: [data.anomaly_count["1"], data.anomaly_count["-1"]],
                    backgroundColor: ["#36A2EB", "#FF6384"]
                }]
            }
        });
    });