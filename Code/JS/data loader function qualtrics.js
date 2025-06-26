// Qualtrics on-load: fetch 'pathoption' and load corresponding scripts
Qualtrics.SurveyEngine.addOnload(function () {
  /**
   * Load the price_series and bots scripts for the given option (1–8).
   *
   * @param {number} option  An integer 1 through 8, selecting one of:
   *   1 → no_risk_start-bottom_high
   *   2 → no_risk_start-bottom_low
   *   3 → no_risk_start-top_high
   *   4 → no_risk_start-top_low
   *   5 → risk_start-bottom_high
   *   6 → risk_start-bottom_low
   *   7 → risk_start-top_high
   *   8 → risk_start-top_low
   */
  function loadCopyTradingScripts(option) {
    const map = {
      1: "no_risk_start-bottom_high",
      2: "no_risk_start-bottom_low",
      3: "no_risk_start-top_high",
      4: "no_risk_start-top_low",
      5: "risk_start-bottom_high",
      6: "risk_start-bottom_low",
      7: "risk_start-top_high",
      8: "risk_start-top_low",
    };

    const suffix = map[option];
    if (!suffix) {
      console.error(`Invalid option "${option}". Must be 1–8.`);
      return;
    }

    const baseURL =
      "https://cdn.jsdelivr.net/gh/benicekh/QualtricsCopyTrading/";
    const files = [
      // price series always at root of Paths
      `Paths/javastrings_price_series_${suffix}.js`,
      // bots under Botdata/
      `Botdata/javastrings_bots_${suffix}.js`,
    ];

    files.forEach((file) => {
      const script = document.createElement("script");
      script.src = baseURL + file;
      script.async = false; // preserve execution order if needed
      document.head.appendChild(script);
    });
  }

  // Load the option value from embedded data
  var option = parseInt(
    Qualtrics.SurveyEngine.getEmbeddedData("pathoption"),
    10
  );

  // Run the loader
  loadCopyTradingScripts(option);
});
