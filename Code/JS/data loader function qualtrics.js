Qualtrics.SurveyEngine.addOnload(function () {
  // 1) Read & parse your embedded data
  var raw = Qualtrics.SurveyEngine.getEmbeddedData("pathoption");
  var option = parseInt(raw, 10);
  if (isNaN(option)) {
    console.error("[CopyTrading] pathoption is not a number:", raw);
    return;
  }

  // 2) Suffix lookup
  var suffixes = [
    null,
    "no_risk_start-bottom_high",
    "no_risk_start-bottom_low",
    "no_risk_start-top_high",
    "no_risk_start-top_low",
    "risk_start-bottom_high",
    "risk_start-bottom_low",
    "risk_start-top_high",
    "risk_start-top_low",
  ];
  var suffix = suffixes[option];
  if (!suffix) {
    console.error("[CopyTrading] Invalid option:", option);
    return;
  }

  // 3) Build your two paths
  var paths = [
    "Paths/javastrings_price_series_" + suffix + ".js",
    "Botdata/javastrings_bots_" + suffix + ".js",
  ];
  var cdnBase =
    "https://cdn.jsdelivr.net/gh/benicekh/QualtricsCopyTrading@main/";
  var rawBase =
    "https://raw.githubusercontent.com/benicekh/QualtricsCopyTrading/main/";

  // 4) Loader via fetch + inline injection
  paths.forEach(function (p) {
    var cdnUrl = cdnBase + p;
    var rawUrl = rawBase + p;

    // Try CDN first
    fetch(cdnUrl)
      .then(function (resp) {
        if (!resp.ok) throw new Error("cdn failed");
        return resp.text();
      })
      .then(function (js) {
        var s = document.createElement("script");
        s.text = js;
        document.head.appendChild(s);
        console.log("[CopyTrading] ✔ Loaded from jsDelivr:", cdnUrl);
      })
      .catch(function () {
        console.warn(
          "[CopyTrading] jsDelivr failed → falling back to raw:",
          cdnUrl
        );
        // Fallback to raw GitHub
        fetch(rawUrl)
          .then(function (resp) {
            if (!resp.ok) throw new Error("raw failed");
            return resp.text();
          })
          .then(function (js) {
            var s = document.createElement("script");
            s.text = js;
            document.head.appendChild(s);
            console.log("[CopyTrading] ✔ Loaded from raw GitHub:", rawUrl);
          })
          .catch(function () {
            console.error("[CopyTrading] ✖ Failed both:", cdnUrl, rawUrl);
          });
      });
  });
});

Qualtrics.SurveyEngine.addOnReady(function () {
  console.log(data);
  console.log(pricePaths);
  Qualtrics.SurveyEngine.setEmbeddedData("botdata", JSON.stringify(data));
  Qualtrics.SurveyEngine.setEmbeddedData(
    "pricepaths",
    JSON.stringify(pricePaths)
  );
});

Qualtrics.SurveyEngine.addOnUnload(function () {
  /*Place your JavaScript here to run when the page is unloaded*/
});
