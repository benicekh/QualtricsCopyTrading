Qualtrics.SurveyEngine.addOnload(function () {
  /*Place your JavaScript here to run when the page loads*/
});

Qualtrics.SurveyEngine.addOnReady(function () {
  /**
   * Created by AFE on 1/27/2020.
   */

  //Uses Fisher-Yates to shuffle a provided array
  function shuffleArray(arr) {
    var j, x, i;
    for (i = arr.length - 1; i > 0; i--) {
      j = Math.floor(Math.random() * (i + 1));
      x = arr[i];
      arr[i] = arr[j];
      arr[j] = x;
    }
    return arr;
  }

  //Returns a random integer between min and max (both inclusive)
  function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  //Samples a random values from given array,
  // Returns as array
  //Note: does not work if array contains not-JSON serializable content
  function sampleRandomElementFromArray(arr) {
    var arrcopy = JSON.parse(JSON.stringify(arr));
    return shuffleArray(arrcopy)[0];
  }

  //Creates a controller for the Markov process. The stock price starts at startprice in a random state.
  // Can have two states: good 70% up, 30% down. Bad is inverted
  // Contains the following functions:
  //  GET_RETURN: returns an object containing two properties: delta (difference in stock price from old, as a ratio) and price (new stock price).
  //      Return is based on state. If good state, return is randomly sampled from [1.03, 1.06, 1.09] (all equally likely) with 70% probably. With remaining 30% is sampled from [.97, .94, .91].
  //      Price is rounded to the nearest integer
  //      Automatically stores the result in an internal array (fetch with function below).
  // UPDATE_STATE: with 80% probability does nothing. 20% probability to change the state.
  // GET_HISTORY: returns an array containing the entire history of the price
  MarkovController = function (Markov_Param) {
    //Randomly determine the initial state
    var state = shuffleArray(["GOOD", "BAD"])[0];

    //Initialize the price at 120
    var price = Markov_Param.startprice;

    //Keep track of the price development
    var history = [price];

    //Keep track of the up-probability and the Bayesian. Call calculateBayes to update.
    var bayes = Markov_Param.prior; //Initial value
    var probability_up =
      Markov_Param.prior * Markov_Param.p_up_good +
      (1 - Markov_Param.prior) * Markov_Param.p_up_bad;

    this.get_next_return = function () {
      var delta = 0;
      //Randomly determine return based on state
      if (state === "GOOD") {
        if (Math.random() < Markov_Param.p_up_good) {
          delta = shuffleArray(Markov_Param.return_high)[0];
        } else {
          delta = shuffleArray(Markov_Param.return_low)[0];
        }
      } else {
        if (Math.random() < Markov_Param.p_up_bad) {
          delta = shuffleArray(Markov_Param.return_high)[0];
        } else {
          delta = shuffleArray(Markov_Param.return_low)[0];
        }
      }

      //Calculate new stockprice
      var oldprice = price;
      price = Math.round(price + delta);

      //Update the history
      history.push(price);

      //Return the object
      return { oldprice: oldprice, delta: delta, newprice: price };
    };

    this.get_history = function () {
      return history;
    };

    this.update_state = function () {
      //Only with 20% probability should the state be changed
      if (Math.random() <= 1 - Markov_Param.p_stay) {
        //State has to be changed
        if (state === "GOOD") {
          state = "BAD";
        } else {
          state = "GOOD";
        }
      }
    };

    //Returns the current state of the asset
    this.getState = function () {
      return state;
    };

    //Updates the Bayesian and up_probabilty.
    // Assumes that the price has already been changed! Hence history[t] = new price, history[t-1] = old price
    this.updateBayes = function () {
      var oldprice = history[history.length - 2];
      if (typeof oldprice === "undefined") {
        oldprice = Markov_Param.startprice;
      }

      var lastStateUpdate;
      if (price > oldprice) {
        lastStateUpdate =
          (Markov_Param.p_up_good * bayes) /
          (Markov_Param.p_up_good * bayes +
            Markov_Param.p_up_bad * (1 - bayes));
      } else {
        lastStateUpdate =
          ((1 - Markov_Param.p_up_good) * bayes) /
          ((1 - Markov_Param.p_up_good) * bayes +
            (1 - Markov_Param.p_up_bad) * (1 - bayes));
      }

      bayes =
        Markov_Param.p_stay * lastStateUpdate +
        (1 - Markov_Param.p_stay) * (1 - lastStateUpdate);
      probability_up =
        bayes * Markov_Param.p_up_good + (1 - bayes) * Markov_Param.p_up_bad;
    };

    //Returns current Bayes value
    this.getBayes = function () {
      return bayes;
    };

    //Returns current probability up
    this.getP_UP = function () {
      return probability_up;
    };

    //Resets the Markov state
    this.reset = function () {
      history = [];
      state = shuffleArray(["GOOD", "BAD"])[0];
      price = Markov_Param.startprice;
      bayes = Markov_Param.prior;
      probability_up =
        Markov_Param.prior * Markov_Param.p_up_good +
        (1 - Markov_Param.prior) * Markov_Param.p_up_bad;
    };
  };

  //A wrapper for the chart controller. Takes the number of rounds as an input (can be changed with the reset function)
  ChartController = function (nrounds, startprice) {
    //Needs a reference to the chart div. This is hardcoded!
    var divname = "chartdiv";

    //Defining plot layout
    var layout = {
      showlegend: false,
      width: "95%",
      height: 300,
      title: {
        text: "Asset Price Development",
        font: {
          size: 15,
        },
      },
      margin: {
        l: 50,
        r: 0,
        b: 50,
        t: 50,
        pad: 0,
      },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      yaxis: {
        title: "Price in ECU",
        titlefont: {
          size: 15,
          color: "black",
        },
        linecolor: "white",
        ticks: "inside",
        tickfont: {
          color: "black",
        },
        showgrid: false,
      },
      xaxis: {
        title: "Period",
        titlefont: {
          size: 15,
          color: "black",
        },
        tickfont: {
          color: "black",
        },
        autotick: false,
        ticks: "outside",
        tick0: 0,
        dtick: 1,
        range: [-0.5, nrounds + 0.5],
        showgrid: false,
      },
      hoverlabel: {
        bgcolor: "white",
        font: { color: "black", size: 20 },
      },
    };

    //Will store the y-axis values
    var y = [startprice];

    //Create a plot data element
    var Plotdata = [
      {
        y: y,
        mode: "lines",
        marker: { color: "blue", size: 10 },
      },
    ];

    //Already draw an empty graph on initialization
    Plotly.plot(divname, Plotdata, layout, { staticPlot: true });

    //Call to plot the price path with a new value
    this.updatePlotPath = function (newprice) {
      Plotdata[0].y.push(newprice);
      setrange();
      Plotly.newPlot(divname, Plotdata, layout, { staticPlot: true });
    };

    //Call to reset the chart entirely. Needs a new number of rounds
    this.reset = function (newrounds) {
      layout.xaxis.range = [-0.5, newrounds + 0.5];
      setrange();

      Plotdata[0].y = [startprice];

      if (newrounds > 20) {
        layout.xaxis.dtick = 10;
      }

      Plotly.newPlot(divname, Plotdata, layout);
    };

    //Sets the range of the graph such that the lines are a bit more centered.
    // Sets the range based on 10 off from the min and max of the return sequence.
    function setrange() {
      //Get the min and max of the plotted array
      var range = Plotdata[0].y;
      var max = Math.max.apply(null, range);
      var min = Math.min.apply(null, range);

      //Set the range in the layout object
      Plotly.relayout(divname, "yaxis.range", [min - 10, max + 10]);
    }

    //Copies the current state of the graph to an object with the given name
    this.copyToNewObject = function (name) {
      Plotly.newPlot(name, Plotdata, layout, { staticPlot: true });
    };
  };

  //Creates the slider and all its interactions. This is done entirely programatically (only needs an empty div to work with), as I want a more generalized version to use later again.
  SliderController = function (
    divname,
    min,
    max,
    step,
    question,
    labels,
    popupbox_enabled,
    indicateMoved
  ) {
    //DEFAULT PARAMETER OBJECT
    var SP = {
      //Functionality options
      //If set to TRUE, this sets a different CSS for the slider after being reset by setSliderValue.
      //This will automatically be changed to the value defined in slider.CSS upon user input.
      //One exception: will also this different CSS on initialization! (As no user input has been received at this point)
      unmovedIndication: indicateMoved,
      unmovedIndicationCSS: {
        // background: "red",
        border: "2px solid darkred",
      },

      //CSS below
      containerCSS: {
        height: "100%",
        textAlign: "center",
        padding: "2%",
      },

      title: {
        label: question,
        CSS: {
          width: "100%",
          fontSize: "normal",
        },
      },

      popupBox: {
        enabled: popupbox_enabled,
        reservedHeight: "90px",
        CSS: {
          width: "100%",
          height: "40px",
          backgroundColor: "white",
          position: "relative",
          zIndex: -1,
        },
        boxCSS: {
          width: "80px",
          height: "30px",
          fontSize: "large",
          borderColor: "2 px solid black",
          borderRadius: "5%",
          backgroundColor: "white",
          border: "2px solid dimgray",
          position: "absolute",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          userSelect: "none",
          zIndex: 10,
        },
        prefix: "",
        postfix: "%",
      },

      slider: {
        min: min,
        max: max,
        step: step,
        class: "customSliderAF",
        CSS: {
          width: "100%",
          height: "5px",
          backgroundColor: "lightgray",
          // borderRadius: "3px",
          border: "1px solid gray",
        },
        CSSthumb: {
          width: "25px",
          height: "25px",
          background: "#2b2b2b",
          cursor: "pointer",
          "border-radius": "100%",
        },
      },

      labels: {
        values: labels,
        paddingTop: "0%",
        CSS: {
          zIndex: -1,
          fontSize: "large",
          display: "inline-block",
          position: "absolute",
          userSelect: "none",
        },
      },
    };

    //Applies a css object to a given element
    function applyCSS(CSSobj, elem) {
      for (key in CSSobj) {
        if (CSSobj.hasOwnProperty(key)) {
          elem.style[key] = CSSobj[key];
        }
      }
    }

    //Creates a title object (P) and appends it to the container. Sets a reference in the ELEMENTS object under "Title"
    function createTitle() {
      //Create object
      var TitleObj = document.createElement("P");

      //Apply the text and CSS
      TitleObj.innerHTML = SP.title.label;
      applyCSS(SP.title.CSS, TitleObj);

      //Append to the container
      Container.appendChild(TitleObj);

      //Store a reference in ELEMENTS
      ELEMENTS.Title = TitleObj;
    }

    //Creates a div to hold the pop-up box
    function createPUBcontainer() {
      var PUBC = document.createElement("DIV");

      //Apply the css
      applyCSS(SP.popupBox.CSS, PUBC);

      //Append to container
      Container.appendChild(PUBC);

      //Store ref
      ELEMENTS.PopUpBoxContainer = PUBC;
    }

    //Creates the slider element.
    function createSlider() {
      var slider = document.createElement("input");
      slider.type = "range";
      slider.min = SP.slider.min;
      slider.max = SP.slider.max;
      slider.step = SP.slider.step;
      slider.classList.add(SP.slider.class);

      //Override default slider css
      slider.style.appearance = "none";
      slider.style.webkitAppearance = "none";
      applyCSS(SP.slider.CSS, slider);

      //Add a custom stylesheet for the thumb. This has to be appended to the header of the document.
      var style = document.createElement("style");

      //Transform the css object into a string.
      var cssstring = "";
      for (key in SP.slider.CSSthumb) {
        cssstring = cssstring + key + ":" + SP.slider.CSSthumb[key] + ";";
      }
      var css =
        "." +
        SP.slider.class +
        "::-webkit-slider-thumb{" +
        "-webkit-appearance: none; appearance: none;" +
        cssstring +
        "}";
      css =
        css + "." + SP.slider.class + "::-moz-range-thumb{" + cssstring + "}";
      css = css + "." + SP.slider.class + ":focus{ outline:none}";

      if (style.styleSheet) {
        style.styleSheet.cssText = css;
      } else {
        style.appendChild(document.createTextNode(css));
      }

      document.getElementsByTagName("head")[0].appendChild(style);

      Container.appendChild(slider);
      ELEMENTS.slider = slider;
    }

    //Creates the labels below the slider element
    function createLabels() {
      //Create a label container
      var LabelContainer = document.createElement("DIV");
      LabelContainer.style.width =
        ELEMENTS.slider.getBoundingClientRect().width + "px";
      LabelContainer.style.paddingTop = SP.labels.paddingTop;
      LabelContainer.style.position = "relative";
      Container.appendChild(LabelContainer);
      ELEMENTS.LabelContainer = LabelContainer;

      //Calculate label width
      var labs = SP.labels.values;
      var len = labs.length;
      var cleft = LabelContainer.getBoundingClientRect().left;
      var cwidth = LabelContainer.getBoundingClientRect().width;

      //Now add all the labels. Store them in ELEMENTS.LABELS
      ELEMENTS.Labels = [];
      for (var i = 0; i < len; i++) {
        //Create label element
        var Lab = document.createElement("DIV");

        //Set text
        Lab.innerHTML = labs[i];

        //Apply css
        applyCSS(SP.labels.CSS, Lab);

        //Set proper left. This should be (cleft) + (i / len)*cwidth
        Lab.style.left = cleft + (i / (len - 1)) * cwidth - i * 2 + "px";
        Lab.style.width = cwidth / len;

        //Store and append
        ELEMENTS.Labels.push(Lab);
        LabelContainer.appendChild(Lab);
      }
    }

    //When the slider's y-dimension is changed, the labels need to be re-positioned. Whenver this happens, call this function
    //TODO: automate
    this.repositionLabels = function () {
      ELEMENTS.LabelContainer.style.width =
        ELEMENTS.slider.getBoundingClientRect().width + "px";

      for (var i = 0; i < ELEMENTS.Labels.length; i++) {
        var labs = SP.labels.values;
        var len = labs.length;
        var cleft = 0; // ELEMENTS.LabelContainer.getBoundingClientRect().left;
        var cwidth =
          0.99 * ELEMENTS.LabelContainer.getBoundingClientRect().width;

        ELEMENTS.Labels[i].style.left =
          cleft + (i / (len - 1)) * cwidth - i * 2 + "px";
        ELEMENTS.Labels[i].style.width = cwidth / len;
        // ELEMENTS.Labels[i].style.width = (cwidth / ELEMENTS.Labels.length) + "px"
      }
    };

    //Call on construction to initialize all elements to the container. Assumes container is already defined
    function setElements() {
      //Applying the CSS to the container
      applyCSS(SP.containerCSS, Container);

      //Container should hold the following objects:
      //  TITLE (if label !== "")
      // Space for the popup-box (if SP.popupbox.enabled)
      // The slider itself
      // Labels (if labels.values.length > 0)

      if (SP.title.label !== "") {
        createTitle();
      }
      if (SP.popupBox.enabled) {
        createPUBcontainer();
      }
      createSlider();
      //If set in the param, set a CSS here to indicate subject has not input a new value
      if (SP.unmovedIndication) {
        applyCSS(SP.unmovedIndicationCSS, ELEMENTS.slider);
      }

      if (SP.labels.values.length > 0) createLabels();
    }

    //Returns the current RELATIVE position of the slider (0 to 1)
    function getSliderPos() {
      var val = ELEMENTS.slider.value;
      var range = SP.slider.max - SP.slider.min;
      return (val - SP.slider.min) / range;
    }

    //Creates a pop-up box, with reference to it being ELEMENTS.popupbox
    function createPUB() {
      var PUB = document.createElement("DIV");
      applyCSS(SP.popupBox.boxCSS, PUB);
      ELEMENTS.popupbox = PUB;

      //Append to the PUBcontainer
      ELEMENTS.PopUpBoxContainer.appendChild(PUB);

      //Set proper location
      changePUBpos();
    }

    //Deletes the pop-up box
    function deletePUB() {
      ELEMENTS.popupbox.parentNode.removeChild(ELEMENTS.popupbox);
    }

    //Changes the x position to the pop-up box
    function changePUBpos() {
      //Get current slider position.
      var pos = getSliderPos();

      //In order to determine the position of the box, we need to know how wide the container is AND how wide the box itself is
      var cw = ELEMENTS.PopUpBoxContainer.getBoundingClientRect().width;
      var bw = ELEMENTS.popupbox.getBoundingClientRect().width;
      var bl = 0; // Math.floor( ELEMENTS.PopUpBoxContainer.getBoundingClientRect().left);

      //So the left property should be pos*cw - 0.5*bw (to center the box in the middle.
      var left = pos * cw - 0.5 * bw;

      //However, left should not be <bw or >(cw - bw)
      if (left < bl) {
        left = bl;
      }
      if (left > cw - bw + bl) {
        left = cw - bw + bl;
      }

      //Set the position
      ELEMENTS.popupbox.style.left = left + "px";

      //Set the text
      ELEMENTS.popupbox.innerHTML =
        SP.popupBox.prefix + ELEMENTS.slider.value + SP.popupBox.postfix;
    }

    //Create a reference to the div containing the slider.
    var Container = document.getElementById(divname);

    //Stores all elements created in the code below
    var ELEMENTS = {};

    //Returns the value of the slider
    this.getSliderValue = function () {
      return ELEMENTS.slider.value;
    };

    //Returns a boolean indicating whether or not the user has moved the slider since its position has been changed
    var sliderMovedBySubject = false;
    this.getSliderMovedBySubject = function () {
      return sliderMovedBySubject;
    };

    //Sets the slider to a new position. Also sets slidermoved to false
    this.setSliderValue = function (val) {
      ELEMENTS.slider.value = val;
      sliderMovedBySubject = false;

      //If set in the param, set a CSS here to indicate subject has not input a new value
      if (SP.unmovedIndication) {
        applyCSS(SP.unmovedIndicationCSS, ELEMENTS.slider);
      }
    };

    //Call whenever the slider is moved
    function sliderMoved() {
      //If the pop-upbox is enabled, move it
      if (SP.popupBox.enabled) {
        if (PUBexists) {
          changePUBpos();
        } else {
          createPUB();
          PUBexists = true;
        }
      }

      //If a different backgroundcolor has been set prior to user input, then set it back to default here
      if (SP.unmovedIndication) {
        applyCSS(SP.slider.CSS, ELEMENTS.slider);
        //Take note that the slider has been moved by the subject
        sliderMovedBySubject = true;
      }
    }
    //Initialize elements
    setElements();

    //To prevent duplicate creations, set to TRUE on creation of a PUB and to FALSE upon deletion.
    var PUBexists = false;

    //set EVENT HANDLERS after initalization
    ELEMENTS.slider.oninput = function () {
      sliderMoved();
    };
    ELEMENTS.slider.onmousedown = function () {
      sliderMoved();
    };
    ELEMENTS.slider.onmouseup = function () {
      //If the pop-upbox is enabled, delete it
      if (SP.popupBox.enabled) {
        deletePUB();
        PUBexists = false;
      }
    };
  };

  //Creates a wrapper for all the qualtrics-specific stuff. Needs the questionID
  QualtricsManager = function () {
    //Next button and output field
    var NextButton = document.getElementById("NextButton");

    //This boolean tests if we're currently in Qualtrics.
    var inQualtrics = NextButton !== null;
    console.warn("Qualtrics enviroment detected: " + inQualtrics);

    //On initalization, hide the nextbutton and output field
    if (inQualtrics) {
      NextButton.style.display = "none";
    }

    //Attempts to paste the output into the
    this.writeData = function (data) {
      if (inQualtrics) {
        Qualtrics.SurveyEngine.setEmbeddedData("Data", JSON.stringify(data));
      } else {
        console.warn("Failed to write data, as we're not in qualtrics");
        console.log(data);
        console.log(JSON.stringify(data));
      }
    };

    //Submits the qualtrics data
    this.submitQuestion = function () {
      if (inQualtrics) {
        NextButton.click();
      } else {
        console.warn("Failed to hit Qualtrics NextButton");
      }
    };

    //Calculates the subjects earnings: one stage wealth and one accuracy bonus
    this.setEmbeddedData = function (expdata) {
      //Select a random stage and round
      var SelectedRound = sampleRandomElementFromArray(expdata.rounds);
      while (
        SelectedRound.stg === "alwaysBuyTraining" ||
        SelectedRound.stg === "alwaysSellTraining" ||
        SelectedRound.r === 1
      ) {
        SelectedRound = sampleRandomElementFromArray(expdata.rounds);
      }

      var SelectedStage = sampleRandomElementFromArray(expdata.stagesummaries);

      //Wealth in ECU: Wealth
      Qualtrics.SurveyEngine.setEmbeddedData(
        "Wealth",
        SelectedStage.wealth_end
      );

      //Payment round: PaymentRound
      Qualtrics.SurveyEngine.setEmbeddedData("PaymentRound", SelectedRound.r);

      //Payment round: StageSelected
      Qualtrics.SurveyEngine.setEmbeddedData("PaymentStage", SelectedRound.ses);

      //Subject forecast PaymentForecast
      var x = (SelectedRound.belief * 100).toFixed(1);
      Qualtrics.SurveyEngine.setEmbeddedData("PaymentForecast", x);

      //True forecast (up_prob): PaymentUpProba
      var y = (SelectedRound.p_up * 100).toFixed(1);
      Qualtrics.SurveyEngine.setEmbeddedData("PaymentTrueProb", y);

      //Absolute error: ForecastError
      Qualtrics.SurveyEngine.setEmbeddedData(
        "ForecastError",
        Math.abs(x - y).toFixed(1)
      );

      //Forecast payment in ECU: ForecastPayment
      Qualtrics.SurveyEngine.setEmbeddedData(
        "ForecastPayment",
        SelectedRound.acc_b
      );

      //Total payment in CHF: Payment
      var totalbonus = SelectedStage.wealth_end + SelectedRound.acc_b;
      var exchangerate = 1 / 20;
      var payoff = (totalbonus * exchangerate * 10) / 10;
      Qualtrics.SurveyEngine.setEmbeddedData("Payment", payoff.toFixed(1));
    };

    //Retrieves a variable from Qualtrics EmbeddedVariables
    this.getEmbeddedDadta = function (variable) {
      if (inQualtrics) {
        console.log(
          "Retrieving variable " +
            variable +
            "at value " +
            Qualtrics.SurveyEngine.getEmbeddedData(variable)
        );
        return Qualtrics.SurveyEngine.getEmbeddedData(variable);
      } else {
        console.warn(
          "Currently not in Qualtrics; unable to retrieve " + variable
        );
      }
    };
  };

  //Creates a controller for the experiment
  ExpController = function (treatment, Markov_Param, QualtricsManager) {
    //Start with an endowment of 170 ECU, a price of 120 ECU and no ownership of the asset
    var cash = Markov_Param.startcash;
    var price = Markov_Param.startprice;
    var price_at_start = price;

    //Keeps track of the current round and whether or not the training period has been completed
    var trainingrounds = 5;
    var rounds = 40;
    var currentround = 0;

    //Set stages, depending on treatment (2 should be different)
    var Stages = [
      { stage: "training", rounds: trainingrounds },
      { stage: "regular", rounds: rounds },
    ];
    if (treatment === 2) {
      Stages = [
        { stage: "alwaysSellTraining", rounds: trainingrounds },
        { stage: "alwaysSell", rounds: rounds },
      ];
    }

    var currentstage = 0;

    //Denotes the asset position
    var ownsAsset = false;

    // The two below are for record-keeping
    var ownsAsset_atStart = false;
    var cash_atStart = cash;
    var actionLastRound = "";

    //Keep track of which price the asset is purchased at. Set to false if the asset is not owned
    var purchasePrice = false;

    //Keeps track of the rounds returns. Should only be set by the feedback screen function
    var roundreturns = {};

    //Keeps track of whether the trade button is toggled on or off
    var tradebuttonON = false;

    //Create a controller for the stock price
    var MC = new MarkovController(Markov_Param);

    //Create a controller for the graph
    var CC = new ChartController(Stages[0].rounds, Markov_Param.startprice);

    //Create a controller for the sliders (but prediction slider only for treatments 1 and 2)
    if (treatment === 1 || treatment === 2) {
      var PredictionSlider = new SliderController(
        "sliderdiv",
        35,
        65,
        0.1,
        "What do you think is the probability that the asset price will <u>increase</u> from now to the next period?",
        [35, 40, 45, 50, 55, 60, 65],
        true,
        true
      );
    }
    var SatisfactionSlider = new SliderController(
      "stage_satisfaction_slider",
      0,
      100,
      1,
      "How satisfied are you with your trading outcome?",
      ["Very unsatisfied", "Neutral", "Very satisfied"],
      false,
      true
    );

    //Keeps track of all the round data in an object. Should have a different array element per round.
    // New elements should be introduced by calling the store_round_data function!
    var DATA = {};
    DATA.rounds = [];
    DATA.stagesummaries = [];

    //Stores all the data for a single round
    function storeDataRound() {
      //Compile round data.
      var stage = Stages[0].stage;

      //loser has a value of 0 if the stock if not owned, 1 if the stock is owned and the price and down and 0 otherwise
      var loser = 0;
      if (ownsAsset) {
        if (roundreturns.delta < 0) {
          loser = 1;
        } else {
          loser = -1;
        }
      }
      //Calculate accuracy and accuracy bonus for this round
      var error, accuracybonus;
      if (treatment !== 3) {
        error = Math.abs(
          PredictionSlider.getSliderValue() / 100 - MC.getP_UP()
        );
        accuracybonus = getAccuracyBonus(error);
      }

      var x = {
        r: currentround,
        stg: stage,
        ses: Math.ceil(currentstage / 2),
        p_s: roundreturns.oldprice,
        p_c: roundreturns.delta,
        p_e: roundreturns.newprice,
        a_s: ownsAsset_atStart,
        a_e: ownsAsset,
        c_s: cash_atStart,
        c_e: cash,
        a_state: MC.getState(),
        bayes: MC.getBayes().toFixed(3),
        p_up: MC.getP_UP().toFixed(3),
      };

      if (treatment !== 3) {
        x.belief = (PredictionSlider.getSliderValue() / 100).toFixed(3);
        x.belief_error = error.toFixed(3);
        x.acc_b = accuracybonus;
      }

      //Save in array.
      DATA.rounds.push(x);
    }

    //Stores the summary data of a stage
    function storeDataStage() {
      var StageWealths = getStageReturns();

      //Push data to stage summary data array
      DATA.stagesummaries.push({
        stage: Stages[0].stage,
        wealth_end: StageWealths.wealth_at_end,
        trade_gains: StageWealths.wealth_difference,
        satisfaction: SatisfactionSlider.getSliderValue(),
      });
    }

    //Save the references to all screen elements that need to be adjusted in the course of the experiment
    var SCREENELEMENTS = {
      //Subscreen elements
      round_title: document.getElementById("round_TITLE"),
      decision_screen: document.getElementById("round_DECISION"),
      feedback_screen: document.getElementById("round_FEEDBACK"),
      satisfaction_screen: document.getElementById("stage_SATISFACTION"),

      //Instruction elements
      instructions: document.getElementById("instructions"),
      instructions_text: document.getElementById("instructions_text"),
      instructions_button: document.getElementById("BUTTON_INSTRUCTIONS"),

      //Decision screen elements
      decision_roundnum: document.getElementById("table_ROUNDNUM"),
      decision_price: document.getElementById("table_PRICE"),
      decision_positiontext: document.getElementById("table_POSITIONTEXT"),
      decision_positionvalue: document.getElementById("table_POSITIONVALUE"),
      decision_shares: document.getElementById("table_SHARES"),
      decision_tradebutton: document.getElementById("table_BUTTON"),
      decision_cash: document.getElementById("table_CASH"),
      decision_nextbutton: document.getElementById("BUTTON_DECISION"),

      //Feedback screen elements
      feedback_delta: document.getElementById("feedback_DELTA"),
      feedback_price: document.getElementById("feedback_NEWPRICE"),
      feedback_position: document.getElementById("feedback_POSITION"),
      feedback_nextbutton: document.getElementById("BUTTON_FEEDBACK"),

      //Satisfaction screen elements
      satisfaction_summary: document.getElementById("stage_summary"),
      satisfaction_button: document.getElementById("BUTTON_SATISFACTION"),
    };

    ///////////////////////
    // CUSTOM FUNCTIONS ///
    ///////////////////////

    //Buys or sells the asset
    function tradeAsset() {
      //If the subject owns the asset, sell it. Otherwise reverse
      if (ownsAsset) {
        //Sell it
        cash = cash + price;
        ownsAsset = false;
        purchasePrice = false;
      } else {
        //Buy it.
        cash = cash - price;
        ownsAsset = true;
        purchasePrice = price;
      }
    }

    //For a given error, returns the per-round accuracy bonus
    function getAccuracyBonus(error) {
      if (error > 0.1) {
        return -20;
      } else {
        if (error >= 0.05) {
          return 0;
        } else {
          if (error >= 0.03) {
            return 20;
          } else {
            if (error >= 0.01) {
              return 40;
            } else {
              return 60;
            }
          }
        }
      }
    }

    //Continues to the feedback screen. ALSO UPDATES PRICE
    function gotoFeedbackScreen() {
      //Purchase or sell the asset if the trade button is toggled
      if (tradebuttonON) {
        tradeAsset();
      }

      //Update the asset price. Store the old price for record keeping
      roundreturns = MC.get_next_return();
      price_at_start = price;
      price = roundreturns.newprice;

      //Update the state
      MC.update_state();

      //Update the stock price and cash amount on the feedback screen
      setFeedbackScreenElements();

      //Hide the decision screen and show the feedback screen
      SCREENELEMENTS.decision_screen.style.display = "none";
      SCREENELEMENTS.feedback_screen.style.display = "inherit";
    }

    //Call to set all the feedback screen elements
    function setFeedbackScreenElements() {
      if (roundreturns.delta < 1) {
        SCREENELEMENTS.feedback_delta.innerHTML =
          "The asset price decreased by " +
          (roundreturns.oldprice - roundreturns.newprice) +
          " ECU";
      } else {
        SCREENELEMENTS.feedback_delta.innerHTML =
          "The asset price increased by " +
          (roundreturns.newprice - roundreturns.oldprice) +
          " ECU";
      }

      SCREENELEMENTS.feedback_price.innerHTML =
        "The new price of the asset is: " + roundreturns.newprice + " ECU";

      if (ownsAsset) {
        var gain = price - purchasePrice; //price - price_at_start;
        if (treatment === 1 || treatment === 3) {
          if (purchasePrice <= price) {
            SCREENELEMENTS.feedback_position.innerHTML =
              "Your current position is at a gain: " + gain + " ECU"; //price-purchasePrice
          } else {
            if (purchasePrice === price) {
              SCREENELEMENTS.feedback_position.innerHTML =
                "Your current position is neither at gain not loss: 0 ECU"; // price-purchasePrice
            } else {
              SCREENELEMENTS.feedback_position.innerHTML =
                "Your current position is at a loss: " + gain + " ECU"; // price-purchasePrice
            }
          }
        } else {
          if (purchasePrice <= price) {
            SCREENELEMENTS.feedback_position.innerHTML =
              "<br>Your asset is automatically sold. <br>Your last trade resulted in a gain: " +
              gain +
              " ECU"; //price-purchasePrice
          } else {
            SCREENELEMENTS.feedback_position.innerHTML =
              "<br>Your asset is automatically sold. <br>Your last trade resulted in a loss: " +
              gain +
              " ECU"; // price-purchasePrice
          }
        }
      } else {
        if (
          Stages[0].stage === "alwaysSell" ||
          Stages[0].stage === "alwaysSellTraining"
        ) {
          SCREENELEMENTS.feedback_position.innerHTML =
            "You did not buy the asset this period";
        }
        if (
          Stages[0].stage === "alwaysBuy" ||
          Stages[0].stage === "alwaysBuyTraining"
        ) {
          SCREENELEMENTS.feedback_position.innerHTML =
            "You sold the asset this period";
        }
        if (Stages[0].stage === "training") {
          SCREENELEMENTS.feedback_position.innerHTML = "";
        }
        if (treatment === 1 || treatment === 3) {
          SCREENELEMENTS.feedback_position.innerHTML =
            "You currently don't own the asset";
        }
      }
    }

    //Goes to the next round
    function next_round() {
      scrollToTop();

      //Update the chart by adding the new price. Again. only if this is not round 0
      if (currentround > 0) {
        CC.updatePlotPath(price);
      }

      if (currentround < Stages[0].rounds) {
        //Store all the data for the just-finished round (assuming this is not round 0, which we use to set up the task)
        if (currentround > 0) {
          //log data for OLD round (the one we just finished)
          storeDataRound();

          //Update the bayes and prop up
          MC.updateBayes();
        }

        //Set the slider back to 50 (if not in treatment 3)
        if (treatment !== 3) {
          PredictionSlider.setSliderValue(50);
        }

        //Keep track of which action was taken this round
        actionLastRound = "none";
        if (ownsAsset !== ownsAsset_atStart) {
          if (ownsAsset) {
            actionLastRound = "bought";
          } else actionLastRound = "sold";
        }

        //THIS IS WHERE THE STAGES DIFFER
        switch (Stages[0].stage) {
          case "alwaysBuy":
            //In this stage, the subject is forced to buy the asset before the next round starts.
            if (!ownsAsset) {
              tradeAsset();
            }
            break;
          case "alwaysBuyTraining":
            //In this stage, the subject is forced to buy the asset before the next round starts.
            if (!ownsAsset) {
              tradeAsset();
            }
            break;
          case "alwaysSell":
            //In this stage the subject is focred to sell the asset
            if (ownsAsset) {
              tradeAsset();
            }
            break;
          case "alwaysSellTraining":
            if (ownsAsset) {
              tradeAsset();
            }
            break;
          case "regular":
            break;
        }

        //Update the current round counter
        currentround++;

        //Keep track of whether the asset is owned at start of round and cash at start of round
        ownsAsset_atStart = ownsAsset;
        cash_atStart = cash;

        //Set the correct value to all decision screen elements, reset the trade button and slider.
        setDecisionScreenElements();

        //Hide the feedback screen and show the decision screen
        SCREENELEMENTS.decision_screen.style.display = "inherit";
        SCREENELEMENTS.feedback_screen.style.display = "none";

        //Reset the slider label positions (excluding treatment 3)
        if (treatment !== 3) {
          setTimeout(function () {
            PredictionSlider.repositionLabels();
          }, 10);
        }
      } else {
        //The last round of this stage is completed, continuing to the satisfaction question if not in training. If we are in training, just continue to the next stage
        if (
          Stages[0].stage === "alwaysBuyTraining" ||
          Stages[0].stage === "alwaysSellTraining" ||
          Stages[0].stage === "training"
        ) {
          nextStage();
        } else {
          showSatisfactionScreen();
        }
      }
    }

    //Call at new round to set the decision screen elements
    function setDecisionScreenElements() {
      //Change the position text, assset price, position value and the text of the trading button. Also unlock the trading button if it was locked before
      SCREENELEMENTS.decision_roundnum.innerHTML = "Period " + currentround;
      SCREENELEMENTS.decision_cash.innerHTML = cash + " ECU";
      SCREENELEMENTS.decision_price.innerHTML = price + " ECU";
      var gain = price - purchasePrice; //price - price_at_start;

      if (treatment === 2) {
        SCREENELEMENTS.decision_positiontext.innerHTML = "";
        SCREENELEMENTS.decision_positionvalue.innerHTML = "";
      } else {
        if (ownsAsset) {
          if (price > purchasePrice) {
            SCREENELEMENTS.decision_positiontext.innerHTML =
              "Your current position is at a gain:";
            SCREENELEMENTS.decision_positionvalue.innerHTML = gain + " ECU";
          } else {
            if (price === purchasePrice) {
              SCREENELEMENTS.decision_positiontext.innerHTML =
                "Your current position is neither at a gain nor at a loss:";
              SCREENELEMENTS.decision_positionvalue.innerHTML = gain + " ECU";
            } else {
              SCREENELEMENTS.decision_positiontext.innerHTML =
                "Your current position is at a loss:";
              SCREENELEMENTS.decision_positionvalue.innerHTML = gain + " ECU";
            }
          }
        } else {
          SCREENELEMENTS.decision_positiontext.innerHTML =
            "You currently don't own the asset";
          SCREENELEMENTS.decision_positionvalue.innerHTML = "";
        }
      }

      //Somewhat dirty and manual tweak for the ab treatment
      /*   if(Stages[0].stage === "alwaysBuy" || Stages[0].stage === "alwaysBuyTraining" ){
            if(actionLastRound !== "none"){
                SCREENELEMENTS.decision_positiontext.innerHTML = "You sold the asset last period";
                SCREENELEMENTS.decision_positionvalue.innerHTML = "";
            }
        }else{
            if(Stages[0].stage === "alwaysSell" || Stages[0].stage === "alwaysSellTraining"){
                if(actionLastRound !== "none"){
                    SCREENELEMENTS.decision_positiontext.innerHTML = "Your gain from last period";
                    SCREENELEMENTS.decision_positionvalue.innerHTML = (gain) + " ECU"
                }
            }
        }*/

      //Unlock the trade button if it has been locked. Also make sure that the contents of the button match the action.
      toggleTradeButtonOff();
      if (ownsAsset) {
        SCREENELEMENTS.decision_tradebutton.innerHTML = "SELL";
      } else {
        SCREENELEMENTS.decision_tradebutton.innerHTML = "BUY";
      }

      //On the first round, hide text on the second line referring to position
      if (currentround === 1) {
        SCREENELEMENTS.decision_positiontext.innerHTML = "";
        SCREENELEMENTS.decision_positionvalue.innerHTML = "";
      }

      //In treatment 3, display the Bayes (we can re-use the sliderdiv here)
      if (treatment === 3) {
        var Sliderdiv = document.getElementById("sliderdiv");
        Sliderdiv.style.textAlign = "left";
        Sliderdiv.innerHTML =
          "The probability for a price increase is " +
          (MC.getP_UP() * 100).toFixed(1) +
          "%";
      }
    }

    //Toggles the trade button to on or off. Does not actually sell or buy the asset!
    function toggleTradeButton() {
      //Visually toggle the button to on or off depending on the current state.
      if (tradebuttonON) {
        tradebuttonON = false;
        SCREENELEMENTS.decision_tradebutton.style.color = "black";
        SCREENELEMENTS.decision_tradebutton.style.backgroundColor = "lightgray";
      } else {
        tradebuttonON = true;
        SCREENELEMENTS.decision_tradebutton.style.color = "white";
        SCREENELEMENTS.decision_tradebutton.style.backgroundColor = "dimgray";
      }
    }

    //Call to force the trade button to off at the start of a round
    function toggleTradeButtonOff() {
      tradebuttonON = false;
      SCREENELEMENTS.decision_tradebutton.style.color = "black";
      SCREENELEMENTS.decision_tradebutton.style.backgroundColor = "#eeeeee";
    }

    //Goes to the next stage. Note that stages are spliced at 0 (old stages are discarded from the Stages array.
    //If there are no more stages to play, call finishMain.
    function nextStage() {
      console.log("Starting Stage" + (currentstage + 1));
      //Load the next round
      if (currentstage !== 0) {
        //Store data for the last round of previous stage first
        storeDataRound();
        Stages.splice(0, 1);
      }
      currentstage++;

      if (Stages.length > 0) {
        //Reset the graph.
        CC.reset(Stages[0].rounds);

        //Reset the round counter, price, cash, roundreturns, ownsAsset, purchasePrice
        cash = Markov_Param.startcash;
        price = Markov_Param.startprice;
        price_at_start = price;
        ownsAsset = false;
        ownsAsset_atStart = false;
        cash_atStart = cash;
        purchasePrice = false;
        currentround = 0;
        roundreturns = {};

        //Reset the Markov controller
        MC.reset();

        //Change the position. If the stage is alwaysBuy, then the subject starts with an asset.
        if (
          Stages[0].stage === "alwaysBuy" ||
          Stages[0].stage === "alwaysBuyTraining"
        ) {
          ownsAsset = true;
          cash = cash - Markov_Param.startprice;
          cash_atStart = cash;
        }
        //Hide the decision screen and feedback screen.
        SCREENELEMENTS.decision_screen.style.display = "none";
        SCREENELEMENTS.feedback_screen.style.display = "none";

        //Show the next set of instructions
        setInstructionText();
      } else {
        finishMain();
      }
    }

    // Call at end of stage (but before the next one starts) to determine the stage's earnings (cash + value of asset if owned).
    // Returns an object containing three values: wealth at start, wealth at end and gains from trading
    function getStageReturns() {
      //Find the first round of the stage
      var FirstRound = {};
      for (var i = 0; i < DATA.rounds.length; i++) {
        if (DATA.rounds[i].stg === Stages[0].stage) {
          FirstRound = DATA.rounds[i];
          break;
        }
      }

      //Calculate wealths
      var w0 = FirstRound.c_s + FirstRound.a_s * FirstRound.p_s;
      var wend = cash + ownsAsset * price;
      var wdiff = wend - w0;

      return {
        wealth_at_start: w0,
        wealth_at_end: wend,
        wealth_difference: wdiff,
      };
    }

    //Show the statisfaction screen after a stage is completed, but before the next starts.
    function showSatisfactionScreen() {
      //Show the satisfaction screen
      SCREENELEMENTS.satisfaction_screen.style.display = "inline-block";

      //Hide the feedback screen
      SCREENELEMENTS.feedback_screen.style.display = "none";

      //Set the earnings text
      var earnings = getStageReturns();
      SCREENELEMENTS.satisfaction_summary.innerHTML =
        "Your total gains from trading are: " +
        earnings.wealth_difference +
        " ECU <br>" +
        "Your final wealth after trading is: " +
        earnings.wealth_at_end +
        " ECU.";

      //Copy the chart
      CC.copyToNewObject("satisfaction_chartdiv");

      //Reset the slider
      SatisfactionSlider.repositionLabels();
      SatisfactionSlider.setSliderValue(50);
    }

    //Finishes the main trading part
    function finishMain() {
      //Write the data to Qualtrics
      QualtricsManager.writeData(DATA);

      //Set embedded payment data
      QualtricsManager.setEmbeddedData(DATA);

      //Submit the qualtrics question
      QualtricsManager.submitQuestion();
    }

    //Sets the instruction text based on the current state (only takes action in ABTraining or ASTraining.
    function setInstructionText() {
      var text = "";
      var title = "";
      /*
      //Text for the start of the actual stages
        if(Stages[0].stage === "alwaysBuy" || Stages[0].stage === "alwaysSell" || Stages[0].stage === "regular"){
            text = "The trial is now completed. <br>" +
                "<br>" +
                "On the next page, you will start with the actual payment relevant rounds of the trading game.";
            title = "Trial Periods Over";
        }else{
            //Instruction text consists of the following segments:
            SCREENELEMENTS.round_title.style.display = "none";

            //  A header (differnt between sessions)
            var header = "You will see a price chart and information on your cash account and the gain or loss from your trading decision in the previous period. <br><br>"
            if(currentstage === 3){
                header = "The first session is over and we now move to the second session. <br><br>"
            }

            //  Type instructions (different between types
            var type = ""
            if(Stages[0].stage === "alwaysBuyTraining"){
                type = "In this session <u>you own the asset at the beginning of each period.</u> " +
                    " You will see a <u>“Sell” button</u> that you can activate to sell the asset." +
                    " Once you click the button, the color of it will turn into dark gray to indicate that it is activated and your sell order will be submitted." +
                    " If you want to revoke your order you can click the button again to deactivate it. <br><br>";
            }
            if(Stages[0].stage === "alwaysSellTraining"){
                type = "In this session <u> you do not own the asset at the beginning of each period.</u> " +
                    " You will see a <u>“Buy” button</u> that you can activate to buy the asset." +
                    " Once you click the button, the color of it will turn into dark gray to indicate that it is activated and your buy order will be submitted. " +
                    " If you want to revoke your order you can click the button again to deactivate it. <br><br>"
            }
            if(Stages[0].stage === "training"){
                type = "On the next page you can play 5 trial periods to get used to the trading environment of the game. <br>" +
                    "<br>" +
                    "You will see a price chart and information on your cash account and the gain or loss of your current holdings in the asset. <br>" +
                    "<br>" +
                    "If you don't own the asset you will see a 'Buy' button that you can activate to buy the asset. Similarly, you will see a 'Sell' button if you own the asset in the current period. " +
                    "Once you click the button, the color of it will turn into a dark gray to indicate that it is activated and your buy or sell order will be submitted. " +
                    "If you want to revoke your order you can click the button again to deactivate it.<br>" +
                    "<br>" +
                    "Once you have made your trading decision and probability estimate, you can move to the next period but clicking the 'Next' button. " +
                    "The 'Next' button will only be activated after you submitted a probability estimate on a slider (if the button is not instantaneously activated you need to click on the slider again). <br>" +
                    "<br>" +
                    "Bear in mind that the <u>trial periods are independent of the payment relevant periods </u> and your decisions in the <u>trial periods have no effect on your payment</u>."
            }

            //  Button instructions (only present in first session)
            var button = "";
            if(currentstage === 1){
                button = "Once you have made your trading decision and probability estimate, you can move to the next period by clicking the 'Next' button. " +
                    "The 'Next' button will only be activated after you submitted a probability estimate on a slider (if the button is not instantaneously activated you need to click on the slider again). <br><br>"
            }

            //Additional type instructions (different between types)
            var type2 = ""
            if(Stages[0].stage === "training"){
                type2= "more text here "
            }
            if(Stages[0].stage === "alwaysBuyTraining"){
                type2 = "<u>If you sold the asset in one period, it will be automatically bought for you at the beginning of the next period. </u>" +
                    "If you want to continue to not invest in the asset you have to sell it again in each period. <br><br>"
            }
            if(Stages[0].stage === "alwaysSellTraining"){
                type2 = "<u>If you bought the asset in one period, it will be automatically sold after your gains or losses for this period are added or deduced from your cash. </u>" +
                    "If you want to continue investing in the asset, you have to buy it again in each period. <br><br>"
            }

            //  A tail (identical)
            var tail = "On the next page you can play 5 trial periods to get used to the trading environment of the game. <br> <br>" +
                "Bear in mind that the <u>trial periods are independent of the payment relevant periods </u> and your decisions in the <u>trial periods have no effect on your payment</u>."

            //Merging the text
            if(Stages[0].stage === "training"){
                text = type
            }else{
                text = header + type + button + type2 + tail
            }

        }*/
      console.log(Stages[0].stage);
      if (
        Stages[0].stage === "training" ||
        Stages[0].stage === "alwaysSellTraining"
      ) {
        switch (treatment) {
          case 1:
            text =
              "On the next page you can play 5 trial periods to get used to the trading environment of the game.<br>" +
              " <br>" +
              "You will see a price chart and information on your cash account and the gain or loss of your current position in the asset.<br>" +
              " <br>" +
              "If you don't own the asset you will see a 'Buy' button that you can activate to buy the asset. Similarly, you will see a 'Sell' button if you own the asset in the current period. Once you click the button, the color of it will turn into a dark gray to indicate that it is activated and your buy or sell order will be submitted. If you want to revoke your order you can click the button again to deactivate it.<br>" +
              " <br>" +
              "Once you have made your trading decision and probability estimate, you can move to the next period by clicking the 'Next' button. The 'Next' button will only be activated after you submitted a probability estimate on a slider. <br>" +
              " <br>" +
              "The trial periods are independent of the payment relevant periods and your decisions in the trial periods have no effect on your payment.<br>";
            break;
          case 2:
            text =
              "You will see a price chart and information on your cash account. <br>" +
              " <br>" +
              "You do not own the asset at the beginning of each period. You will see a “Buy” button that you can activate to buy the asset. Once you click the button, the color of it will turn into dark gray to indicate that it is activated and your buy order will be submitted. If you want to revoke your order you can click the button again to deactivate it.<br>" +
              " <br>" +
              "Once you have made your trading decision and probability estimate, you can move to the next period by clicking the 'Next' button. The 'Next' button will only be activated after you submitted a probability estimate on a slider. <br>" +
              " <br>" +
              "If you bought the asset in one period, it will be automatically sold after your gains or losses for this period are added or deducted from your cash. If you want to continue investing in the asset, you have to buy it again in each period.<br>" +
              " <br>" +
              "On the next page you can play 5 trial periods to get used to the trading environment of the game.<br>" +
              " <br>" +
              "The trial periods are independent of the payment relevant periods and your decisions in the trial periods have no effect on your payment.<br>";
            break;
          case 3:
            text =
              "On the next page you can play 5 trial periods to get used to the trading environment of the game.<br>" +
              " <br>" +
              "You will see a price chart and information on your cash account and the gain or loss of your current position in the asset.<br>" +
              " <br>" +
              "If you don't own the asset you will see a 'Buy' button that you can activate to buy the asset. Similarly, you will see a 'Sell' button if you own the asset in the current period. Once you click the button, the color of it will turn into a dark gray to indicate that it is activated and your buy or sell order will be submitted. If you want to revoke your order you can click the button again to deactivate it.<br>" +
              " <br>" +
              "Once you have made your trading decision, you can move to the next period by clicking the 'Next' button. <br>" +
              " <br>" +
              "The trial periods are independent of the payment relevant periods and your decisions in the trial periods have no effect on your payment.";
            break;
        }
      }

      if (Stages[0].stage === "regular" || Stages[0].stage === "alwaysSell") {
        title = "Trial periods over";
        text =
          "The trial is now completed. <br>" +
          "<br> On the next page, you will start with the actual payment relevant rounds of the trading game.";
      }

      //Setting the text and title
      SCREENELEMENTS.round_title.innerHTML = title;
      SCREENELEMENTS.instructions_text.innerHTML = text;

      //Showing instructions
      SCREENELEMENTS.instructions.style.display = "inline-block";
    }

    //Call to scroll to the top of the page
    function scrollToTop() {
      document.body.scrollTop = document.documentElement.scrollTop = 0;
    }

    //EVENT LISTENERS
    //The button used to leave the instructions (one button for all sets of instructions)
    SCREENELEMENTS.instructions_button.onclick = function () {
      //Hide the instruction text
      SCREENELEMENTS.instructions.style.display = "none";

      //Set the "Trial Rounds" title in triaining periods
      if (
        Stages[0].stage === "alwaysBuy" ||
        Stages[0].stage === "alwaysSell" ||
        Stages[0].stage === "regular"
      ) {
        SCREENELEMENTS.round_title.style.display = "none";
      } else {
        SCREENELEMENTS.round_title.innerHTML = "TRIAL PERIODS";
        SCREENELEMENTS.round_title.style.display = "inline-block";
      }

      //Start next round
      next_round();
    };

    //The button used to trade assets
    SCREENELEMENTS.decision_tradebutton.onclick = function () {
      toggleTradeButton();
    };

    //The next button at the decision screen
    SCREENELEMENTS.decision_nextbutton.onclick = function () {
      if (treatment === 3) {
        gotoFeedbackScreen();
      } else {
        if (PredictionSlider.getSliderMovedBySubject()) {
          gotoFeedbackScreen();
        }
      }
    };

    //The next button at the feedback screen
    SCREENELEMENTS.feedback_nextbutton.onclick = function () {
      next_round();
    };

    //The next button on the satisfaction screen (starts the next stage)
    SCREENELEMENTS.satisfaction_button.onclick = function () {
      if (SatisfactionSlider.getSliderMovedBySubject()) {
        SCREENELEMENTS.satisfaction_screen.style.display = "none";
        storeDataStage();
        nextStage();
      }
    };

    //On creation, show the instructions
    //setInstructionText();
    nextStage();

    //Holds the link and text to instructions

    //Appends the correct link at the bottom of the page
    document.getElementById("instructions_link").onclick = function () {
      let link = InstructionLinks["treatment" + treatment];
      window.open(link, "_blank");
    };
  };

  var MarkovParameters = {
    startprice: 125, //Starting price of the asset
    startcash: 200, //Cash amount at start
    p_stay: 0.85, //Probability of the asset to NOT switch states
    return_high: [2, 4, 6], //Set of good returns
    return_low: [-2, -4, -6], //Set of bad returns
    p_up_good: 0.75, //Probability of a good return in the GOOD state.
    p_up_bad: 0.25, //Probability of a good return in the BAD state.
    prior: 0.5, //Initial prior
  };

  var InstructionLinks = {
    treatment1:
      "https://www.dropbox.com/s/i5vefgpg069omgk/_1_Instructions%20Baseline.pdf?dl=0",
    treatment2:
      "https://www.dropbox.com/s/paqezlhijpcmsb9/_2_Instructions%20Automated%20Selling.pdf?dl=0",
    treatment3:
      "https://www.dropbox.com/s/ijlfdbkyi1cq0ev/_3_Instructions%20Bayes.pdf?dl=0",
  };

  var QM = new QualtricsManager();
  var treatment = parseInt(QM.getEmbeddedDadta("treatment"));
  console.log(treatment);
  var EC = new ExpController(treatment, MarkovParameters, QM);

  //Treatments: 1 = Baseline, 2= Always Sell, 3 = True Bayes
});

Qualtrics.SurveyEngine.addOnUnload(function () {
  /*Place your JavaScript here to run when the page is unloaded*/
});
