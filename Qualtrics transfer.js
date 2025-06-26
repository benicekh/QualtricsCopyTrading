Qualtrics.SurveyEngine.addOnload(function ()
//hides the next button on a page
this.hidePreviousButton();
jQuery('#Plug').attr('style', 'display:none !important');
});

Qualtrics.SurveyEngine.addOnReady(function () {
  let treatment;

  treatment = parseInt(Qualtrics.SurveyEngine.getEmbeddedData("treatment"));

  let pathseries;
  pathseries = Qualtrics.SurveyEngine.getEmbeddedData("pathseries");

  let pricePaths;
  pricePaths = JSON.parse(Qualtrics.SurveyEngine.getEmbeddedData("pricepaths"));

  let dataTraining;
  dataTraining = JSON.parse(
    Qualtrics.SurveyEngine.getEmbeddedData("dataTraining")
  );
  let data;
  data = JSON.parse(Qualtrics.SurveyEngine.getEmbeddedData("botdata"));
  let followersTreatment;
  followersTreatment = parseInt(
    Qualtrics.SurveyEngine.getEmbeddedData("followersTreatment")
  );
  let TLs
  TLs = JSON.parse(Qualtrics.SurveyEngine.getEmbeddedData("TLs"));
	

	//draw round that pays out
	let stageDrawn = Math.floor(Math.random() * 10);
let winningIndex = stageDrawn + 1;
console.log(winningIndex);

// Function to toggle column visibility and adjust colspan
function toggleColumnBasedOnTreatment() {
  const columns = document.querySelectorAll(".toggle-column");
  const topPerformersCell = document.getElementById("table_TOPPERFORMERS");

  if (followersTreatment === 0) {
    columns.forEach((column) => column.classList.add("hidden"));
    topPerformersCell.setAttribute("colspan", "7");
  } else {
    columns.forEach((column) => column.classList.remove("hidden"));
    topPerformersCell.setAttribute("colspan", "8");
  }
}

// Call the function to apply the initial state
toggleColumnBasedOnTreatment();	


  let nameTreatment;
  nameTreatment = parseInt(
    Qualtrics.SurveyEngine.getEmbeddedData("nameTreatment")
  );

  console.log(data["stage_0"]["round_0"]);

  //document.getElementById("copydata").classList.add("hidden");

  document.getElementById("rounddata").classList.add("hidden");

  document.getElementById("chartdiv").classList.add("hidden");

  document.getElementById("chartwrap").classList.add("hidden");

  document.getElementById("copydata").classList.add("hidden");

  // Initialize Variables

  /// Price Path Version

  let pathVersion;

  let pathPrice;

  let path;

  let copiedTL = null;

  /// Phase played 0 - training, rest decisions

  let phase = 0;

  /// max phases

  let maxPhases;

  /// random price path or fixed order

  const randompath = 0;

  //// Starting Round data

  var roundDataStart = {
    price: 250, //Price of the asset
    cash: 2500, //Cash amount
    asset: 0, // assets held
    round: 0, // current round
    portfolio: 0, // unrealised asset value
    rankingClicks: [], // use of buttons in ranking
    plotClicks: [], // use of buttons to see strategy
  };

  var roundDataPersistent = {
    gain: 0, // total cash gain all phases

    endowments: 0, // total endowments so far for returns calculation

    return: 0, // total return all phases
  };

  //// Round data\

  var roundData = {
    price: roundDataStart.price, //Price of the asset
    cash: roundDataStart.cash, //Cash amount
    asset: roundDataStart.asset, // assets held
    round: roundDataStart.round, // current round
    portfolio: roundDataStart.portfolio, // unrealised asset value
    rankingClicks: roundDataStart.rankingClicks, // use of buttons in ranking
    plotClicks: roundDataStart.plotClicks, // use of buttons to see strategy
    next: 0, // count of button use
    previous: 0,
  };

  //// Next Round data

  var roundDataNew = roundData;

  /// Rounds/stage

  const trainingrounds = 5;

  var rounds = 41;
  var currentround = roundDataStart.round;
  /// Stages object Training/Treatment
  const Stages = [
    { stage: "training", rounds: trainingrounds },
    { stage: "regular", rounds: rounds },
  ];

  /// Function to randomly pick price path

  //// set pathPrice to relevant chosen price path array

  // Random integer

 const getRandomInt = function (min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

/// Random or sequential picking of price path
const pathPicker = function () {
  if (randompath == 1) {
    // Random
    var num = getRandomInt(0, Object.keys(pricePaths).length - 1);
    var path = Object.keys(pricePaths)[num];
  } else {
    var num = 0;
    var path = Object.keys(pricePaths)[num];
  }

  pathPrice = pricePaths[Object.keys(pricePaths)[num]];
  pathVersion = Object.keys(pricePaths)[num];
  console.log(path);
  delete pricePaths[Object.keys(pricePaths)[num]];
  console.log(pricePaths);
  return pathPrice;
};

var randomProperty = function (obj) {
  var keys = Object.keys(obj);

  return obj[keys[(keys.length * Math.random()) << 0]];
};

//let test = randomProperty(pricePaths);
//console.log(test);

if (Stages[0].stage === "training") {
  pathPrice = [250, 245, 240, 230, 245];
  pathVersion = "training";
} else {
  pathPicker();
}

//pathPrice = pricePaths[pathPicker()];
//console.log(pathPrice);

///// Current
var priceParameters = {
  price: roundData.stockPrice, //Price of the asset
  cash: roundData.cash, //Cash amount
};
///// Main data storage, to be passed to Qualtrics
var DATA = {};
DATA.rounds = [];
DATA.stagesummaries = [];

DATA.roundSeries = [];
DATA.priceSeries = [];
DATA.assetsSeries = [];
DATA.OngoingReturn = [];
const storeDataRound = function () {
  //Compile round data.
  var stage = Stages[0].stage;
  var realValue = roundData.asset * roundData.price;
  var portfolio = returnsCalc(roundData.portfolio, realValue);
  var ongoingReturn = returnsCalc(roundDataStart.cash, wealthCalc());
  DATA.roundSeries.push(roundData.round);
  DATA.priceSeries.push(roundData.price);
  DATA.assetsSeries.push(roundData.asset);
  DATA.OngoingReturn.push(parseInt(ongoingReturn));
  var x = {
    r: roundData.round,
    phase: Stages[0].stage,
    stg: phase,
    p: roundData.price,
    a: roundData.asset,
    c: roundData.cash,
    path: pathVersion,
    portfolio: portfolio,
    ongoingReturn: ongoingReturn,
    unrealized: roundData.portfolio,
    clicks: roundData.rankingClicks,
    plotclicks: roundData.plotClicks,
    next: roundData.next,
    previous: roundData.previous,
    treatment: treatment,
    pathseries: pathseries,
  };
  roundData.next = 0;
  roundData.previous = 0;

  //Save in array.
  DATA.rounds.push(x);
};

// References to screen elements
var SCREENELEMENTS = {
  //Subscreen elements
  round_title: document.getElementById("round_TITLE"),
  decision_screen: document.getElementById("round_DECISION"),
  copy_data_screen: document.getElementById("copydata"),
  instructions: document.getElementById("instructions"),
  instructions_nextbutton: document.getElementById("instruction_next"),
  instructions_text: document.getElementById("instruction_text"),
  /*   //Instruction elements
  instructions: document.getElementById("instructions"),
  instructions_text: document.getElementById("instructions_text"),
  instructions_button: document.getElementById("BUTTON_INSTRUCTIONS"), */

  //Decision screen elements
  decision_roundnum: document.getElementById("table_ROUNDNUM"),
  decision_price: document.getElementById("table_PRICE"),
  decision_positiontext: document.getElementById("table_POSITIONTEXT"),
  decision_positionvalue: document.getElementById("table_POSITIONVALUE"),
  decision_shares: document.getElementById("table_SHARES"),
  decision_buybutton: document.getElementById("table_BUTTON_BUY"),
  decision_sellbutton: document.getElementById("table_BUTTON_SELL"),
  decision_cash: document.getElementById("table_CASH"),
  decision_nextbutton: document.getElementById("BUTTON_DECISION"),
  decision_nextbutton_training: document.getElementById(
    "BUTTON_TRAINING_START"
  ),
  decision_nextbutton_show_info: document.getElementById("BUTTON_INFO_SHOW"),
  decision_shares_label: document.getElementById("label_SHARES"),
  decision_price_label: document.getElementById("label_PRICE"),
  decision_cash_label: document.getElementById("label_CASH"),
  decision_return_label: document.getElementById("label_RETURN"),
  decision_return: document.getElementById("table_RETURN"),
  copytable: document.getElementById("copytable"),

  //End of Round Screen
  eor: document.getElementById("table_ENDOFROUND"),
  eorwealth: document.getElementById("table_EORWEALTH"),
  eorcash: document.getElementById("table_EORCASH"),
  eorreturn: document.getElementById("table_EORRETURN"),
  eorcashall: document.getElementById("table_EORCASHALL"),
  eorreturnall: document.getElementById("table_EORRETURNALL"),
  eorwealthall: document.getElementById("table_EORWEALTHALL"),
  // Player list
  copytable_header: document.getElementById("table_TOPPERFORMERS"),
  rank1: document.getElementById("table_RANK1"),
  player1: document.getElementById("table_PLAYER1"),
  wealth1: document.getElementById("table_PLAYER1_WEALTH"),

  return1: document.getElementById("table_PLAYER1_RETURN"),
  wealthall1: document.getElementById("table_PLAYER1_WEALTHALL"),

  retall1: document.getElementById("table_PLAYER1_RETALL"),
  copiers1: document.getElementById("table_PLAYER1_COPIERS"),
  rank2: document.getElementById("table_RANK2"),
  player2: document.getElementById("table_PLAYER2"),
  wealth2: document.getElementById("table_PLAYER2_WEALTH"),

  return2: document.getElementById("table_PLAYER2_RETURN"),
  wealthall2: document.getElementById("table_PLAYER2_WEALTHALL"),

  retall2: document.getElementById("table_PLAYER2_RETALL"),
  copiers2: document.getElementById("table_PLAYER2_COPIERS"),
  rank3: document.getElementById("table_RANK3"),
  player3: document.getElementById("table_PLAYER3"),
  wealth3: document.getElementById("table_PLAYER3_WEALTH"),

  return3: document.getElementById("table_PLAYER3_RETURN"),
  wealthall3: document.getElementById("table_PLAYER3_WEALTHALL"),

  retall3: document.getElementById("table_PLAYER3_RETALL"),
  copiers3: document.getElementById("table_PLAYER3_COPIERS"),
  rank4: document.getElementById("table_RANK4"),
  player4: document.getElementById("table_PLAYER4"),
  wealth4: document.getElementById("table_PLAYER4_WEALTH"),

  return4: document.getElementById("table_PLAYER4_RETURN"),
  wealthall4: document.getElementById("table_PLAYER4_WEALTHALL"),

  retall4: document.getElementById("table_PLAYER4_RETALL"),
  copiers4: document.getElementById("table_PLAYER4_COPIERS"),
  rank5: document.getElementById("table_RANK5"),
  player5: document.getElementById("table_PLAYER5"),
  wealth5: document.getElementById("table_PLAYER5_WEALTH"),

  return5: document.getElementById("table_PLAYER5_RETURN"),
  wealthall5: document.getElementById("table_PLAYER5_WEALTHALL"),

  retall5: document.getElementById("table_PLAYER5_RETALL"),
  copiers5: document.getElementById("table_PLAYER5_COPIERS"),
  copy_button1: document.getElementById("BUTTON_COPY_1"),
  copy_button2: document.getElementById("BUTTON_COPY_2"),
  copy_button3: document.getElementById("BUTTON_COPY_3"),
  copy_button4: document.getElementById("BUTTON_COPY_4"),
  copy_button5: document.getElementById("BUTTON_COPY_5"),
  copy_next: document.getElementById("BUTTON_BROWSE_NEXT"),
  copy_prev: document.getElementById("BUTTON_BROWSE_PREV"),

  hide_copy: document.getElementById("BUTTON_DECISION_HIDE"),

  // Copy request line
  copy_request: document.getElementById("copyRequest"),
};

// Debug mode hide elements
const hideElement = function (element) {
  element.classList.toggle("hidden");
};

// for testing purposes; comment next line out to reveal button
SCREENELEMENTS.hide_copy.classList.add("hidden");

SCREENELEMENTS.hide_copy.onclick = function () {
  hideElement(SCREENELEMENTS.copytable);
  if (SCREENELEMENTS.copytable.classList.contains("hidden")) {
    SCREENELEMENTS.hide_copy.textContent = "Show Copy Window";
  } else {
    SCREENELEMENTS.hide_copy.textContent = "Hide Copy Window";
  }
};

// Visuals Initialization
/// Hide Qualtrics next button
/// Set Variable screenelements to starting value
const initialize = function () {
  roundData.price = roundDataStart.price;
  roundData.asset = roundDataStart.asset;
  roundData.cash = roundDataStart.cash;
  roundData.round = roundDataStart.round;
  roundData.portfolio = roundDataStart.portfolio;

  DATA.roundSeries = [];
  DATA.priceSeries = [];
  DATA.assetsSeries = [];
  DATA.OngoingReturn = [];
  SCREENELEMENTS.decision_sellbutton.classList.add("unavailable");
  SCREENELEMENTS.decision_nextbutton_show_info.classList.add("hidden");
  if (Stages[0].stage === "regular") {
    document.getElementById("rounddata").classList.remove("hidden");
    document.getElementById("chartdiv").classList.remove("hidden");
    document.getElementById("chartwrap").classList.remove("hidden");
    //document.getElementById("copydata").classList.add("hidden");
    document.getElementById("resultdata").classList.add("hidden");
    SCREENELEMENTS.copytable_header.textContent = "Traders";
    SCREENELEMENTS.decision_nextbutton.classList.remove("hidden");
    SCREENELEMENTS.decision_nextbutton_training.classList.add("hidden");
  } else {
    SCREENELEMENTS.instructions.classList.remove("hidden");
    SCREENELEMENTS.decision_sellbutton.classList.add("unavailable");
    document.getElementById("rounddata").classList.add("hidden");
    document.getElementById("chartdiv").classList.add("hidden");
    document.getElementById("chartwrap").classList.add("hidden");
    //document.getElementById("copydata").classList.add("hidden");
    document.getElementById("resultdata").classList.add("hidden");
    SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
    SCREENELEMENTS.decision_nextbutton_training.classList.add("hidden");
    SCREENELEMENTS.copy_request.classList.add("hidden");
    //SCREENELEMENTS.copytable.classList.add("hidden");
    if (Stages[0].stage === "training") {
      SCREENELEMENTS.instructions_text.textContent =
        "The start button will initiate the training round that will allow you to familiarize yourself with the interface!";
    } else {
      SCREENELEMENTS.instructions_text.textContent =
        "Pressing the start button will begin the trading game!";
    }
  }
  if (treatment == 1) {
    SCREENELEMENTS.copy_data_screen.classList.add("hidden");
    SCREENELEMENTS.copy_request.classList.add("hidden");
  } else if (treatment == 2) {
    console.log(treatment);
    SCREENELEMENTS.copy_button1.classList.add("hidden");
    SCREENELEMENTS.copy_button2.classList.add("hidden");
    SCREENELEMENTS.copy_button3.classList.add("hidden");
    SCREENELEMENTS.copy_button4.classList.add("hidden");
    SCREENELEMENTS.copy_button5.classList.add("hidden");
    SCREENELEMENTS.copy_request.classList.add("hidden");
  } else {
    SCREENELEMENTS.decision_buybutton.classList.add("hidden");
    SCREENELEMENTS.decision_sellbutton.classList.add("hidden");
  }

  if (Stages[0].stage === "training" && roundData.round === 0) {
    maxpage =
      Math.ceil(Object.keys(dataTraining["stage_0"]["round_0"]).length / 5) - 1;
    console.log("maxpage:" + maxpage);
  } else {
    maxpage = Math.ceil(Object.keys(data["stage_0"]["round_0"]).length / 5) - 1;
    console.log("maxpage:" + maxpage);
  }
  page = 0;
  SCREENELEMENTS.copy_prev.classList.add("hidden");
  SCREENELEMENTS.copy_next.classList.remove("hidden");
  if (maxpage === 0) {
    SCREENELEMENTS.copy_next.classList.add("hidden");
  }
};

/// Update Screenelements
const update = function () {
  if (roundData.round !== Stages[0].rounds && treatment != 1) {
    copyMechanism(copiedTL, phase, roundData.round);
  }
  SCREENELEMENTS.decision_price.textContent = roundData.price;
  SCREENELEMENTS.decision_shares.textContent = roundData.asset;
  SCREENELEMENTS.decision_cash.textContent = roundData.cash;
  SCREENELEMENTS.decision_roundnum.textContent =
    "Period: " + (roundData.round + 1);
  if (roundData.round + 1 === Stages[0].rounds) {
    SCREENELEMENTS.decision_roundnum.textContent = "End of Phase";
  }
  var realValue = roundData.asset * roundData.price;
  //SCREENELEMENTS.decision_return.textContent =
  //  returnsCalc(roundData.portfolio, realValue) + "%";
  SCREENELEMENTS.decision_return.textContent = realValue + roundData.cash;
  if (roundData.price === 0) {
    SCREENELEMENTS.decision_buybutton.classList.add("unavailable");
  }
};

// Functions

/// Picking price path randomly

/// Buying Asset

//// Button visuals & event listener

SCREENELEMENTS.decision_buybutton.onclick = function () {
  console.log("buy");

  if (roundData.price > roundData.cash) {
    //// Check for sufficient cash
    console.log("not enough money");
    SCREENELEMENTS.decision_buybutton.classList.add("unavailable"); // change button color on hover
  } else if (roundData.price === 0) {
    //// Check if asset has crashed
    console.log("asset is worthless");
    SCREENELEMENTS.decision_buybutton.classList.add("unavailable"); // change button color on hover
  } else {
    //// Change values - cash, assets held
    roundData.asset++;
    roundData.cash = roundData.cash - roundData.price;
    roundData.portfolio = roundData.portfolio + roundData.price;
    SCREENELEMENTS.decision_sellbutton.classList.remove("unavailable");
  }
  update();
};

/// Selling Asset

SCREENELEMENTS.decision_sellbutton.onclick = function () {
  console.log("sell");

  if (roundData.asset <= 0) {
    //// Check whether assets are owned
    console.log("no assets owned");
    SCREENELEMENTS.decision_sellbutton.classList.add("unavailable"); // change button color on hover
  } else {
    //// Change values - cash, assets held
    roundData.portfolio =
      roundData.portfolio - roundData.portfolio / roundData.asset;
    roundData.asset--;
    roundData.cash = roundData.cash + roundData.price;
  }
  if (
    roundData.asset > 0 &&
    roundData.cash + roundData.price > roundData.price
  ) {
    SCREENELEMENTS.decision_buybutton.classList.remove("unavailable");
  }
  if (roundData.asset == 0) {
    SCREENELEMENTS.decision_sellbutton.classList.add("unavailable");
  }
  update();
};

let TLrank;
/// Next Round
SCREENELEMENTS.decision_nextbutton.onclick = function () {
  Plotly.purge(eorchartdiv);
  //console.log(copyRank);
  //console.log(copiedTL);
  if (
    roundData.round + 1 === Stages[0].rounds &&
    treatment === 3 &&
    copiedTL === null
  ) {
    SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
  }
  if (roundData.round + 1 === Stages[0].rounds) {
    //console.log("Done!");
    finalRound();
  } else if (roundData.round === Stages[0].rounds) {
    //console.log("next round");
    nextPhase();
    TLrank = copyMechanism(copiedTL, phase, roundData.round);
    document.getElementById("copydata").classList.remove("hidden");
    toggleColumnVisibility(true);
  } else if (roundData.round + 2 === Stages[0].rounds) {
    SCREENELEMENTS.decision_nextbutton.textContent = "End Phase";
    endRound();
  } else {
    //console.log("Next!");
    document.getElementById("copydata").classList.add("hidden");
    toggleColumnVisibility(false);
    SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
    endRound();
  }
};

////Hiding the Pre Training Info window
SCREENELEMENTS.decision_nextbutton_training.onclick = function () {
  Plotly.purge(eorchartdiv);
  SCREENELEMENTS.decision_sellbutton.classList.add("unavailable");
  document.getElementById("rounddata").classList.remove("hidden");
  document.getElementById("chartdiv").classList.remove("hidden");
  document.getElementById("chartwrap").classList.remove("hidden");
  //document.getElementById("copydata").classList.add("hidden");
  document.getElementById("resultdata").classList.add("hidden");
  SCREENELEMENTS.copytable_header.textContent = "Traders";
  SCREENELEMENTS.decision_nextbutton.classList.remove("hidden");
  SCREENELEMENTS.decision_nextbutton_training.classList.add("hidden");
  update();
};

/// Hiding the Instruction Window
var originalDisplay = document.querySelector(".centered-container").style
  .display;
SCREENELEMENTS.instructions_nextbutton.onclick = function () {
  document.querySelector(".centered-container").style.display = "none";
  SCREENELEMENTS.instructions.classList.add("hidden");
  SCREENELEMENTS.instructions_nextbutton.classList.add("hidden");
  SCREENELEMENTS.instructions_text.classList.add("hidden");

  if (Stages[0].stage === "training" && roundData.round === 0) {
    if (treatment == 2) {
      SCREENELEMENTS.copytable_header.textContent =
        "Performance of other players during the previous phase is shown here";
      document.getElementById("copydata").classList.remove("hidden");
      toggleColumnVisibility(true);
      SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
      SCREENELEMENTS.decision_nextbutton_training.classList.remove("hidden");
      SCREENELEMENTS.copy_request.classList.add("hidden");
      updateRanks();
    } else if (treatment == 3) {
      SCREENELEMENTS.copytable_header.textContent =
        "Performance of copyable players during the previous phase is shown here";
      document.getElementById("copydata").classList.remove("hidden");
      toggleColumnVisibility(true);
      SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
      SCREENELEMENTS.decision_nextbutton_training.classList.add("hidden");
      SCREENELEMENTS.copy_request.classList.remove("hidden");
      updateRanks();
    } else {
      document.getElementById("rounddata").classList.remove("hidden");
      document.getElementById("chartdiv").classList.remove("hidden");
      document.getElementById("chartwrap").classList.remove("hidden");
      SCREENELEMENTS.decision_nextbutton.classList.remove("hidden");
    }
  } else if (Stages[0].stage === "training" && roundData.round > 0) {
    if (treatment == 2) {
      document.getElementById("copydata").classList.remove("hidden");
      toggleColumnVisibility(true);
      document.getElementById("copytable").style.display = "table";
      SCREENELEMENTS.decision_nextbutton.classList.remove("hidden");
      SCREENELEMENTS.decision_nextbutton_training.classList.add("hidden");
      SCREENELEMENTS.copy_request.classList.add("hidden");
      updateRanks();
    } else if (treatment == 3) {
      document.getElementById("copydata").classList.remove("hidden");
      toggleColumnVisibility(true);
      document.getElementById("copytable").style.display = "table";
      SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
      SCREENELEMENTS.decision_nextbutton_training.classList.add("hidden");
      SCREENELEMENTS.copy_request.classList.remove("hidden");
      updateRanks();
    } else {
      //SCREENELEMENTS.decision_nextbutton.classList.remove("hidden");
      nextPhase();
    }
  }
};

/// Move from training results to copy window for 1st real phase, show info window in between
SCREENELEMENTS.decision_nextbutton_show_info.onclick = function () {
  copiedTL = null;
  copyRank = null;

  document.querySelector(".centered-container").style.display = "block";

  SCREENELEMENTS.instructions.classList.remove("hidden");
  SCREENELEMENTS.instructions_text.classList.remove("hidden");
  document.getElementById("resultdata").classList.add("hidden");
  SCREENELEMENTS.decision_nextbutton_show_info.classList.add("hidden");
  SCREENELEMENTS.instructions_text.textContent =
    "Pressing the start button will begin the trading game!";
  SCREENELEMENTS.instructions_nextbutton.classList.remove("hidden");
};

// End of round function
const endRound = function () {
  // Safe round data
  storeDataRound();
  // Update Plot element
  //CC.updatePlotPath(roundData.price);
  roundData.round++;
  roundData.price = pathPrice[roundData.round];
  CC.updatePlotPath(roundData.price);
  update();
  // Define a function to handle the next button click
  function handleNextButton() {
    if (roundData.round + 1 === Stages[0].rounds) {
      finalRound();
    } else if (roundData.round + 2 === Stages[0].rounds) {
      SCREENELEMENTS.decision_nextbutton.classList.remove("hidden");
      SCREENELEMENTS.decision_nextbutton.textContent = "End Phase";
      endRound();
    } else {
      endRound();
    }
  }

  // Check if another round is needed
  if (roundData.round + 2 <= Stages[0].rounds) {
    // Schedule the next button click after a delay
    setTimeout(handleNextButton, 250);
  }
};

function handleNextButtonClick() {
  //console.log(copyRank);
  //console.log(copiedTL);
  if (
    roundData.round + 1 === Stages[0].rounds &&
    treatment === 3 &&
    copiedTL === null
  ) {
    SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
  }
  if (roundData.round + 1 === Stages[0].rounds) {
    //console.log("Done!");
    finalRound();
  } else if (roundData.round === Stages[0].rounds) {
    //console.log("next round");
    nextPhase();
  } else if (roundData.round + 2 === Stages[0].rounds) {
    SCREENELEMENTS.decision_nextbutton.textContent = "End Phase";
    endRound();
  } else {
    //console.log("Next!");

    endRound();
  }
}

// Final round
const finalRound = function () {
  console.log("final round triggers");
  console.log("Phase" + phase);
  console.log("Round" + roundData.round);
  // Safe round data
  storeDataRound();
  // Update Plot element
  //CC.updatePlotPath(roundData.price);
  roundData.price = pathPrice[roundData.round];
  update();
  updateRanks();
  //wealthCalc();
  // keep track of total gain and return
  if (Stages[0].stage === "regular") {
    SCREENELEMENTS.eor.textContent = "End of Phase: " + (phase + 1);
    roundDataPersistent.gain =
      roundDataPersistent.gain + wealthCalc() - roundDataStart.cash;
    roundDataPersistent.endowments =
      roundDataPersistent.endowments + roundDataStart.cash;
    roundDataPersistent.return = returnsCalc(
      roundDataPersistent.endowments,
      roundDataPersistent.gain + roundDataPersistent.endowments
    );
    if (treatment === 1) {
      document.getElementById("copydata").classList.add("hidden");
      toggleColumnVisibility(false);
    } else {
      document.getElementById("copydata").classList.remove("hidden");
      toggleColumnVisibility(true);
    }
  } else {
    console.log("triggered");
    SCREENELEMENTS.eor.textContent = "End of Training Phase";
    updateRanks();
    SCREENELEMENTS.decision_nextbutton_show_info.classList.remove("hidden");
    SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
    document.getElementById("copydata").classList.add("hidden");
    toggleColumnVisibility(false);
    //document.getElementById("copytable").classList.add("hidden");
    SCREENELEMENTS.copy_request.classList.add("hidden");
  }
  console.log(roundDataPersistent);
  // update Display to show final wealth instead
  SCREENELEMENTS.eorwealth.textContent = wealthCalc();
  SCREENELEMENTS.eorreturn.textContent = returnsCalc(
    roundDataStart.cash,
    wealthCalc()
  );
  SCREENELEMENTS.eorcash.textContent = wealthCalc() - roundDataStart.cash;
  SCREENELEMENTS.eorcashall.textContent = roundDataPersistent.gain;
  SCREENELEMENTS.eorreturnall.textContent = roundDataPersistent.return;
  SCREENELEMENTS.eorwealthall.textContent =
    roundDataPersistent.gain + roundDataPersistent.endowments;

  if (Stages[0].stage === "training") {
    SCREENELEMENTS.eorcashall.textContent = wealthCalc() - roundDataStart.cash;
    SCREENELEMENTS.eorreturnall.textContent = returnsCalc(
      roundDataStart.cash,
      wealthCalc()
    );
    SCREENELEMENTS.eorwealthall.textContent = wealthCalc();
  }
  // hide normal screens and show end of round screen
  document.getElementById("rounddata").classList.add("hidden");
  document.getElementById("chartdiv").classList.add("hidden");
  document.getElementById("chartwrap").classList.add("hidden");
  document.getElementById("resultdata").classList.remove("hidden");

  if (treatment === 3 && copiedTL === null && Stages[0].stage === "regular") {
    SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
    SCREENELEMENTS.copy_request.classList.remove("hidden");
  }
  if (treatment === 1 && Stages[0].stage === "regular") {
    SCREENELEMENTS.copytable.classList.add("hidden");
  }
  if (Stages[0].stage === "training") {
    SCREENELEMENTS.decision_nextbutton.textContent = "Next Phase";
  } else {
    if (phase + 1 < maxPhases) {
      SCREENELEMENTS.decision_nextbutton.textContent = "Next Phase";
    } else {
      SCREENELEMENTS.decision_nextbutton.textContent = "End Game";
    }
  }
  console.log(roundData.plotClicks);
  var x = {
    phaseName: Stages[0].stage,
    phase: phase,
    wealth: wealthCalc(),
    gain: wealthCalc() - roundDataStart.cash,
    phaseReturn: returnsCalc(roundDataStart.cash, wealthCalc()),
    wealthALL: (phase + 1) * roundDataStart.cash + roundDataPersistent.gain,
    gainAll: roundDataPersistent.gain,
    returnAll: roundDataPersistent.return,
    tradeLeader: copiedTL,
    TLrank: TLrank,
    treatment: treatment,
    roundSeries: DATA.roundSeries,
    priceSeries: DATA.priceSeries,
    assetsSeries: DATA.assetsSeries,
    ongoingReturn: DATA.OngoingReturn,
    plotclicks: roundData.plotClicks,
    rankingClicks: roundData.rankingClicks,
	pathseries: pathseries,
    path: pathVersion,
	nameTreatment: nameTreatment, //1 shows risk categories
    followersTreatment: followersTreatment, //0 hides followers
  };
  console.log("pushing stagesummaries");
  console.log(x);
  console.log(roundData.plotClicks);
  DATA.stagesummaries.push(x);
  roundData.rankingClicks = [];
  roundData.plotClicks = [];
  console.log(roundData.plotClicks);
  roundData.round++;
};

// Calculate wealth in final round
const wealthCalc = function () {
  const wealth = roundData.price * roundData.asset + roundData.cash;
  return wealth;
};

// Calculate percentage return between two values
const returnsCalc = function (initial, final) {
  var ret = ((final - initial) / initial) * 100;
  ret = ret.toFixed(2);
  //console.log(ret);
  if (isNaN(ret)) {
    ret = "0.00";
    return ret;
  } else {
    return ret;
  }
};

/// Next Phase

const nextPhase = function () {
  SCREENELEMENTS.decision_nextbutton.textContent = "Next";
  if (Stages[0].stage === "training") {
    Stages.splice(0, 1); // remove old stage from object
    initializePhase();
  } else {
    if (phase + 1 < maxPhases) {
      phase++;
      console.log(DATA);
      initializePhase();
    } else {
      SCREENELEMENTS.decision_nextbutton.textContent = "End Game";
      //console.log("the game is over!");
      //console.log(DATA);
      QM.writeData(DATA);
      let gainValue = (DATA.stagesummaries && DATA.stagesummaries[winningIndex] && 	DATA.stagesummaries[winningIndex].gain) || null;
      console.log(`Winning Index: ${winningIndex}, Gain: ${gainValue}`);
      Qualtrics.SurveyEngine.setEmbeddedData("FinalGain", gainValue);
      Qualtrics.SurveyEngine.setEmbeddedData("WinningRound", winningIndex);
      Qualtrics.SurveyEngine.setEmbeddedData("Endowment", roundDataStart.cash);
      Qualtrics.SurveyEngine.setEmbeddedData(
        "Payout",
        roundDataStart.cash + gainValue
      );
      Qualtrics.SurveyEngine.setEmbeddedData(
        "nameList",
        JSON.stringify(scrambledNameList)
      );

      //QM.writeData(DATA.stagesummaries);
      QM.submitQuestion();
    }
    console.log(Stages[0].stage);
    console.log(Stages[0].rounds);
    console.log("Phase:");
    console.log(phase);
    //
  }
  // Resetting everything
};

const initializePhase = function () {
  // initializing a new phase to be called after last round
  console.log("Phase:");
  console.log(phase);
  initialize();
  update();
  CC.reset(Stages[0].rounds - 1);
  //CC.updatePlotPath(roundData.price);store
  SCREENELEMENTS.decision_price_label.textContent = "Asset Price:";
  SCREENELEMENTS.decision_cash_label.textContent = "Cash:";
  SCREENELEMENTS.decision_shares_label.textContent = "Shares:";
  SCREENELEMENTS.decision_sellbutton.classList.add("exBUTTON--unavailable");
  SCREENELEMENTS.decision_return_label.textContent = "Wealth:";
  SCREENELEMENTS.decision_return.textContent = " 0";
  if (treatment == 1) {
    SCREENELEMENTS.copy_data_screen.classList.add("hidden");
  } else if (treatment == 2) {
    SCREENELEMENTS.copy_button1.classList.add("hidden");
    SCREENELEMENTS.copy_button2.classList.add("hidden");
    SCREENELEMENTS.copy_button3.classList.add("hidden");
    SCREENELEMENTS.copy_button4.classList.add("hidden");
    SCREENELEMENTS.copy_button5.classList.add("hidden");
  } else {
    SCREENELEMENTS.decision_buybutton.classList.add("hidden");
    SCREENELEMENTS.decision_sellbutton.classList.add("hidden");
  }
  pathPicker();
};

//// Check which stage
//// move to next stage, reset everything
//// if finished wrapping up game TO ADD STEPS

// Misc

// Plotly
const ChartController = function (nrounds, startprice) {
  // Needs a reference to the chart div. This is hardcoded!
  var divname = "chartdiv";

  // Defining plot layout
  var layout = {
    showlegend: false,
    width: "95%",
    height: 300,
    title: {
      text: "Share Price Development",
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
      showgrid: true,
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
      tick0: 1,
      dtick: 1,
      range: [0.5, nrounds + 0.5],
      tickvals: Array.from(
        { length: Math.ceil(nrounds / 10) + 1 },
        (_, i) => i * 10 + 1
      ).filter((v) => v <= nrounds),
      showgrid: false,
    },
    hoverlabel: {
      bgcolor: "white",
      font: { color: "black", size: 20 },
    },
  };

  // Will store the y-axis values
  var y = [startprice];

  // Create a plot data element
  var Plotdata = [
    {
      y: y,
      mode: "lines",
      marker: { color: "blue", size: 10 },
    },
  ];

  // Already draw an empty graph on initialization
  Plotly.plot(divname, Plotdata, layout, { staticPlot: true });

  // Call to plot the price path with a new value
  this.updatePlotPath = function (newprice) {
    Plotdata[0].y.push(newprice);
    setrange();
    Plotly.newPlot(divname, Plotdata, layout, { staticPlot: true });
  };

  // Call to reset the chart entirely. Needs a new number of rounds
  this.reset = function (newrounds) {
    layout.xaxis.range = [0.5, newrounds + 0.5];
    layout.xaxis.tickvals = Array.from(
      { length: Math.ceil(newrounds / 10) + 1 },
      (_, i) => i * 10 + 1
    ).filter((v) => v <= newrounds);
    setrange();

    Plotdata[0].y = [startprice];

    Plotly.newPlot(divname, Plotdata, layout);
  };

  // Sets the range of the graph such that the lines are a bit more centered.
  // Sets the range based on 10 off from the min and max of the return sequence.
  function setrange() {
    // Get the min and max of the plotted array
    var range = Plotdata[0].y;
    var max = Math.max.apply(null, range);
    var min = Math.min.apply(null, range);

    // Set the range in the layout object
    Plotly.relayout(divname, "yaxis.range", [min - 10, max + 10]);
  }

  // Copies the current state of the graph to an object with the given name
  this.copyToNewObject = function (name) {
    Plotly.newPlot(name, Plotdata, layout, { staticPlot: true });
  };
};

var CC = new ChartController(Stages[0].rounds - 1, roundData.price);

//Plotly Popup chart controller
const PopupChartController = function (
  nrounds,
  roundseries,
  rightseries,
  leftseries,
  chart,
  title,
  righttitle,
  type,
  startrange,
  name
) {
  //Needs a reference to the chart div. This is hardcoded!
  roundData.rankingClicks.push(name);
  var divname = chart;
  //Defining plot layout
  var layout = {
    showlegend: true,
    width: 350,
    height: 200,
    title: {
      text: title,
      font: {
        size: 10,
      },
    },
    margin: {
      l: 40,
      r: 40,
      b: 80,
      t: 40,
      pad: 5,
      autoexpand: false,
    },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    yaxis: {
      title: "Price in ECU",
      titlefont: {
        size: 10,
        color: "black",
      },
      linecolor: "black",
      ticks: "outside",
      tickfont: {
        color: "black",
      },
      showgrid: true,
      zeroline: false,
      overlaying: "y2",
    },
    yaxis2: {
      title: righttitle,
      titlefont: {
        size: 10,
        color: "black",
      },
      linecolor: "black",
      ticks: "inside",
      tickfont: {
        color: "black",
      },
      showgrid: true,
      zeroline: false,
      side: "right",
    },

    xaxis: {
      title: "Period",
      titlefont: {
        size: 10,
        color: "black",
      },
      tickfont: {
        color: "black",
      },
      autotick: false,
      ticks: "inside",
      //tick0: 0,
      dtick: 1,
      range: [startrange, nrounds - 1 + 0.5],
      zeroline: false,
      showgrid: false,
    },
    hoverlabel: {
      bgcolor: "white",
      font: { color: "black", size: 20 },
    },
    legend: {
      orientation: "h", // Horizontal orientation
      yanchor: "top",
      y: -0.5, // Position below the plot; adjust as needed
      xanchor: "center",
      x: 0.5, // Centered horizontally
    },
  };

  var right = {
    //x: roundseries,
    y: rightseries,
    name: righttitle,
    xaxis: "x",
    yaxis: "y2",
    type: type,
  };
  var left = {
    x: roundseries,
    y: leftseries,
    name: "Price in ECU",
    type: "line",
    xaxis: "x",
    yaxis: "y",
  };
  var data = [left, right];
  Plotly.newPlot(divname, data, layout, { staticPlot: true });
};

//Plotly.plot(divname, Plotdata, layout, { staticPlot: true });

// Qualtrics Interactions

const QualtricsManager = function () {
  //Next button and output field
  var NextButton = document.getElementById("NextButton");

  //This boolean tests if we're currently in Qualtrics.
  var inQualtrics = NextButton !== null;
  console.warn("Qualtrics enviroment detected: " + inQualtrics);

  //On initalization, hide the nextbutton and output field, load treatment and path
  if (inQualtrics) {
    treatment = parseInt(Qualtrics.SurveyEngine.getEmbeddedData("treatment"));
    if (treatment == 1) {
      maxPhases = 11;
    } else {
      maxPhases = 10;
    }
    console.log("max phases: " + maxPhases);
    console.log("treatment: " + treatment);
    console.log(typeof treatment);
    pathseries = Qualtrics.SurveyEngine.getEmbeddedData("pathseries");
    console.log(pathseries);
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
    Qualtrics.SurveyEngine.setEmbeddedData("Wealth", SelectedStage.wealth_end);

    //Payment round: PaymentRound
    Qualtrics.SurveyEngine.setEmbeddedData("PaymentRound", SelectedRound.r);

    //Payment round: StageSelected
    Qualtrics.SurveyEngine.setEmbeddedData("PaymentStage", SelectedRound.ses);

    //Total payment in CHF: Payment
    var totalbonus = SelectedStage.wealth_end;
    var exchangerate = 1 / 20;
    var payoff = (totalbonus * exchangerate * 10) / 10;
    Qualtrics.SurveyEngine.setEmbeddedData("Payment", payoff.toFixed(1));
  };

  //Retrieves a variable from Qualtrics EmbeddedVariables
  this.getEmbeddedData = function (variable) {
    if (inQualtrics) {
      console.log(
        "Retrieving variable " +
          variable +
          " at value " +
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

/* const pp = {};
for (let i = 0; i < 30; i += 1) {
  let propName = `path${i}`; // create a dynamic property name
  pp[propName] = [];
  pp[propName].push(QM.getEmbeddedData(propName));

  //for (let j = 0; j < 40; j += 1) {
  //  let p = `${i}.${j}`;
  //  pp[propName].push(QM.getEmbeddedData(p));
  //}
}*/

/// Top performer functionality

//// Page browsing button
/// calculate number of pages
let page = 0;
let maxpage;

console.log(maxpage);
/// Eventlistner
SCREENELEMENTS.copy_next.onclick = function () {
  if (page + 1 <= maxpage) {
    page += 1;
    roundData.rankingClicks.push("Next");
    roundData.next += 1;
  }
  if (page != 0) {
    SCREENELEMENTS.copy_prev.classList.remove("hidden");
  }
  SCREENELEMENTS.copy_prev.textContent = "Previous 5";
  if (page == maxpage) {
    SCREENELEMENTS.copy_next.classList.add("hidden");
  }
  updateRanks();
};
SCREENELEMENTS.copy_prev.onclick = function () {
  if (page >= 0) {
    page -= 1;
    roundData.rankingClicks.push("Previous");
    roundData.previous += 1;
  }
  if (page == 0) {
    SCREENELEMENTS.copy_prev.classList.add("hidden");
  }
  if (page <= maxpage) {
    SCREENELEMENTS.copy_next.classList.remove("hidden");
  }
  updateRanks();
};

//Function that adjusts displayed playernames depending on risk level and treatment

//Array shuffle for the non informative names
//Run once at startup and save list to Qualtrics
const nameList = ["Trader A", "Trader B", "Trader C", "Trader D", "Trader E"];

function shuffleArray(array) {
  // Create a copy of the array to avoid mutating the original array
  let arr = array.slice();

  // Loop through the array
  for (let i = arr.length - 1; i > 0; i--) {
    // Generate a random index
    const j = Math.floor(Math.random() * (i + 1));

    // Swap elements at index i and index j
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }

  return arr;
}

function processName(crra) {
  // Split the input string by "_"
  const parts = crra.split("_");
  //console.log(parts);
  // Extract the value of X
  const x = parseFloat(parts[1]);
  //console.log(x);

  // Depending on the value of X, return different strings
  // By treatment
  if (Stages[0].stage != "training" || roundData.round > 0) {
    if (nameTreatment === 1) {
      switch (x) {
        case -1.5:
          return "Risk-seeking Trader";
        case 0:
          return "Risk-neutral Trader";
        case 1:
          return "Slightly risk-averse Trader";
        case 3:
          return "Moderately risk-averse Trader";
        case 6:
          return "Highly risk-averse Trader";
        default:
          return "Error";
      }
    } else {
      switch (x) {
        case -1.5:
          return scrambledNameList[0];
        case 0:
          return scrambledNameList[1];
        case 1:
          return scrambledNameList[2];
        case 3:
          return scrambledNameList[3];
        case 6:
          return scrambledNameList[4];
        default:
          return "Error";
      }
    }
  } else {
    return crra;
  }
}

// Rank number
// Updating values
const updateRanks = function () {
  if (treatment != 1) {
    if (Stages[0].stage === "training" && roundData.round === 0) {
      maxpage =
        Math.ceil(Object.keys(dataTraining["stage_0"]["round_0"]).length / 5) -
        1;
    } else {
      maxpage =
        Math.ceil(Object.keys(data["stage_0"]["round_0"]).length / 5) - 1;
    }
    if (maxpage === 0) {
      SCREENELEMENTS.copy_next.classList.add("hidden");
    }
    SCREENELEMENTS.rank1.textContent = 1 + page * 5 + ".";
    SCREENELEMENTS.rank2.textContent = 2 + page * 5 + ".";
    SCREENELEMENTS.rank3.textContent = 3 + page * 5 + ".";
    SCREENELEMENTS.rank4.textContent = 4 + page * 5 + ".";
    SCREENELEMENTS.rank5.textContent = 5 + page * 5 + ".";

    let stage = "stage_" + phase;
    let round = "round_" + 0;
    let rank1 = "rank_" + (1 + page * 5);
    let rank2 = "rank_" + (2 + page * 5);
    let rank3 = "rank_" + (3 + page * 5);
    let rank4 = "rank_" + (4 + page * 5);
    let rank5 = "rank_" + (5 + page * 5);

    for (let i = 1; i <= 5; i++) {
      const playerKey = "player" + i;
      const wealthKey = "wealth" + i;
      const gainKey = "gain" + i;
      const returnKey = "return" + i;
      const wealthallKey = "wealthall" + i;
      const gainallKey = "gainall" + i;
      const retallKey = "retall" + i;
      const copiersKey = "copiers" + i;
      let rankid = "rank_" + i;
      let playerData;

      if (Stages[0].stage === "training" && roundData.round === 0) {
        playerData = dataTraining[stage][round]["rank_" + (i + page * 5)] || {}; // Ensure playerData is an object
      } else if (Stages[0].stage === "training" && roundData.round > 3) {
        playerData = data[stage]["round_0"]["rank_" + (i + page * 5)] || {}; // Ensure playerData is an object
      } else if (Stages[0].stage != "training" && roundData.round === 0) {
        stage = "stage_" + phase;
        playerData = data[stage]["round_0"]["rank_" + (i + page * 5)] || {}; // Ensure playerData is an object
        console.log(stage);
      } else if (
        Stages[0].stage != "training" &&
        phase === maxPhases - 1 &&
        roundData.round === 40
      ) {
        console.log("phase:" + phase);
        console.log("round" + roundData.round);
        stage = "stage_" + phase;
        console.log("stage term: " + stage);
        playerData = data[stage]["round_40"]["rank_" + (i + page * 5)] || {};
      } else {
        console.log("phase:" + phase);
        console.log("round" + roundData.round);
        stage = "stage_" + (phase + 1);
        playerData = data[stage]["round_0"]["rank_" + (i + page * 5)] || {}; // Ensure playerData is an object
        console.log(stage);
      }
      //console.log(playerData);
      const playerValue = playerData["ResponseId"] || "";
      const wealthValue =
        playerData["phaseWealth"] === 0 ? "0" : playerData["phaseWealth"] || "";
      //console.log(wealthValue);
      const returnValue =
        playerData["phaseReturn"] === 0 ? "0" : playerData["phaseReturn"] || "";
      const gainValue =
        playerData["gain"] === 0 ? "0" : playerData["gain"] || "";
      const retallValue =
        playerData["returnAllv2"] === 0 ? "0" : playerData["returnAllv2"] || "";
      const wealthallValue =
        playerData["wealthALL"] === 0 ? "0" : playerData["wealthALL"] || "";
      const currWealthValue =
        playerData["currWealth"] === 0 ? "0" : playerData["currWealth"] || "";
      const gainallValue =
        playerData["gainAll"] === 0 ? "0" : playerData["gainAll"] || "";
      if (followersTreatment == 1) {
        const copiersValue = TLs[nameTreatment][stage][rankid];
        console.log("Copiers Value:");
        console.log(copiersValue);
        console.log(copiersKey);
        SCREENELEMENTS[copiersKey].textContent = copiersValue + "%";
      }

      SCREENELEMENTS[playerKey].textContent = processName(playerValue);
      SCREENELEMENTS[wealthKey].textContent = wealthValue;
      SCREENELEMENTS[returnKey].textContent = returnValue + "%";
      SCREENELEMENTS[wealthallKey].textContent =
        wealthallValue - currWealthValue;
      SCREENELEMENTS[retallKey].textContent = retallValue + "%";
    }

    let playerData1;
    let playerData2;
    let playerData3;
    let playerData4;
    let playerData5;

    if (Stages[0].stage === "training" && roundData.round === 0) {
      playerData1 = dataTraining[stage][round][rank1] || {}; // Ensure playerData is an object
      playerData2 = dataTraining[stage][round][rank2] || {};
      playerData3 = dataTraining[stage][round][rank3] || {};
      playerData4 = dataTraining[stage][round][rank4] || {};
      playerData5 = dataTraining[stage][round][rank5] || {};
    } else if (Stages[0].stage === "training" && roundData.round > 3) {
      playerData1 = data[stage]["round_0"][rank1] || {}; // Ensure playerData is an object
      playerData2 = data[stage]["round_0"][rank2] || {};
      playerData3 = data[stage]["round_0"][rank3] || {};
      playerData4 = data[stage]["round_0"][rank4] || {};
      playerData5 = data[stage]["round_0"][rank5] || {};
    } else if (
      Stages[0].stage != "training" &&
      phase === maxPhases - 1 &&
      roundData.round === 40
    ) {
      stage = "stage_" + phase;
      playerData1 = data[stage][round][rank1] || {}; // Ensure playerData is an object
      playerData2 = data[stage][round][rank2] || {};
      playerData3 = data[stage][round][rank3] || {};
      playerData4 = data[stage][round][rank4] || {};
      playerData5 = data[stage][round][rank5] || {};
    } else {
      stage = "stage_" + (phase + 1);
      playerData1 = data[stage][round][rank1] || {}; // Ensure playerData is an object
      playerData2 = data[stage][round][rank2] || {};
      playerData3 = data[stage][round][rank3] || {};
      playerData4 = data[stage][round][rank4] || {};
      playerData5 = data[stage][round][rank5] || {};
    }

    //console.log(playerData1);
    window.lastPhaseReturn2 = playerData1["phaseReturn"];
    window.lastPhaseReturn3 = playerData2["phaseReturn"];
    window.lastPhaseReturn4 = playerData3["phaseReturn"];
    window.lastPhaseReturn5 = playerData4["phaseReturn"];
    window.lastPhaseReturn6 = playerData5["phaseReturn"];

    window.allPhaseReturn2 = playerData1["returnAllv2"];
    window.allPhaseReturn3 = playerData2["returnAllv2"];
    window.allPhaseReturn4 = playerData3["returnAllv2"];
    window.allPhaseReturn5 = playerData4["returnAllv2"];
    window.allPhaseReturn6 = playerData5["returnAllv2"];

    window.allPhaseGain2 = playerData1["gainAll"];
    window.allPhaseGain3 = playerData2["gainAll"];
    window.allPhaseGain4 = playerData3["gainAll"];
    window.allPhaseGain5 = playerData4["gainAll"];
    window.allPhaseGain6 = playerData5["gainAll"];

    window.thisPhaseReturn2 = returnsCalc(
      roundDataStart.cash - playerData1["unrealized"],
      playerData1["c"]
    );
    window.thisPhaseReturn3 = returnsCalc(
      roundDataStart.cash - playerData2["unrealized"],
      playerData2["c"]
    );
    window.thisPhaseReturn4 = returnsCalc(
      roundDataStart.cash - playerData3["unrealized"],
      playerData3["c"]
    );
    window.thisPhaseReturn5 = returnsCalc(
      roundDataStart.cash - playerData4["unrealized"],
      playerData4["c"]
    );
    window.thisPhaseReturn6 = returnsCalc(
      roundDataStart.cash - playerData5["unrealized"],
      playerData5["c"]
    );

    window.roundseries2 = playerData1["roundSeries"];
    window.roundseries3 = playerData2["roundSeries"];
    window.roundseries4 = playerData3["roundSeries"];
    window.roundseries5 = playerData4["roundSeries"];
    window.roundseries6 = playerData5["roundSeries"];

    window.priceseries2 = playerData1["priceSeries"];
    window.priceseries3 = playerData2["priceSeries"];
    window.priceseries4 = playerData3["priceSeries"];
    window.priceseries5 = playerData4["priceSeries"];
    window.priceseries6 = playerData5["priceSeries"];

    window.returnseries2 = playerData1["ongoingReturnSeries"];
    window.returnseries3 = playerData2["ongoingReturnSeries"];
    window.returnseries4 = playerData3["ongoingReturnSeries"];
    window.returnseries5 = playerData4["ongoingReturnSeries"];
    window.returnseries6 = playerData5["ongoingReturnSeries"];

    window.assetseries2 = playerData1["assetseries"];
    window.assetseries3 = playerData2["assetseries"];
    window.assetseries4 = playerData3["assetseries"];
    window.assetseries5 = playerData4["assetseries"];
    window.assetseries6 = playerData5["assetseries"];

    window.playerid2 = playerData1["ResponseId"];
    window.playerid3 = playerData2["ResponseId"];
    window.playerid4 = playerData3["ResponseId"];
    window.playerid5 = playerData4["ResponseId"];
    window.playerid6 = playerData5["ResponseId"];

    window.player1 = playerData1;
    window.player2 = playerData2;
    window.player3 = playerData3;
    window.player4 = playerData4;
    window.player5 = playerData5;

    document.querySelectorAll(".trader-button").forEach((button, index) => {
      //const traderId = `player${index + 1}`;
      const traderId = "player" + (index + 1);
      console.log(traderId);
      const traderInfo = window[traderId];
      //button.textContent = processName(traderInfo["ResponseId"]); // Update button title to trader's name - removed
      button.textContent = "Show";
    });

    document
      .querySelectorAll(".trader-button-perf")
      .forEach((button, index) => {
        //const traderId = `player${index + 1}`;
        const traderId = "player" + (index + 1);
        console.log(traderId);
        const traderInfo = window[traderId];
        //button.textContent = processName(traderInfo["ResponseId"]); // Update button title to trader's name - removed
        button.textContent = "Show";
      });

    //console.log("Copied TL: ");
    //console.log(copiedTL);
    setButtonClasses(processName(copiedTL));
    hideUnusedButtons("copytable");
  }
};
function hideMessageRow() {
  const messageRow = document.getElementById("messageRow");
  if (messageRow) {
    messageRow.style.display = "none"; // Hide the row
  }
}

function calculateGlobalMaxAssets(traders) {
  // Flatten all asset series arrays from each trader and find the max value
  const allAssetsValues = traders.flatMap((trader) => trader["assetsSeries"]);
  globalMaxAssets = Math.max(...allAssetsValues) * 1.1; // Add 10% padding
}

function calculateGlobalReturn(traders) {
  // Flatten all asset series arrays from each trader and find the max value
  const allReturnValues = traders.flatMap(
    (trader) => trader["ongoingReturnSeries"]
  );
  globalMaxReturn = Math.max(...allReturnValues) * 1.1; // Add 10% padding
  globalMinReturn = Math.min(...allReturnValues) * 1.1;
}

let lastTraderId = null;
let lastChartType = null;
let globalMaxAssets = 0;
let globalMaxReturn = 0;
let globalMinReturn = 0;
// Plotly setup function for chart (this updates the plot)
  function updateChart(trader) {
    const priceSeries = trader["priceSeries"];
    const roundSeries = trader["roundSeries"];
    const secondarySeries = trader["assetseries"]; // Second series for y-axis 2

    const traders = [
      window.player1,
      window.player2,
      window.player3,
      window.player4,
      window.player5,
    ];
    calculateGlobalMaxAssets(traders);

    // Main trace for the primary y-axis
  const trace1 = {
    x: roundSeries,
    y: priceSeries,
    mode: "lines+markers",
    type: "scatter",
    name: "Price",
    yaxis: "y2",
  };

  // Secondary trace for the secondary y-axis
  const trace2 = {
    x: roundSeries,
    y: secondarySeries,
    mode: "lines+markers",
    type: "scatter",
    name: "Assets",
    yaxis: "y1", // Specify the second y-axis
  };
    let head = processName(trader["ResponseId"]) + " Strategy";
  const layout = {
    title: head,
    xaxis: { title: "Rounds" },
    yaxis: {
      title: "Assets",
      side: "left",
      range: [0, globalMaxAssets],
    },
    height: 300,
    yaxis2: {
      title: "Price",
      overlaying: "y",
      side: "right",
    },
    legend: {
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: -0.4,
    },
  };
  const config = {
    displayModeBar: false, // this is the line that hides the bar.
  };
  Plotly.newPlot("eorchartdiv", [trace1, trace2], layout, config);
}

  function updateChartPerf(trader) {
    const priceSeries = trader["priceSeries"];
    const roundSeries = trader["roundSeries"];
    const secondarySeries = trader["ongoingReturnSeries"]; // Second series for y-axis 2

    const traders = [
      window.player1,
      window.player2,
      window.player3,
      window.player4,
      window.player5,
    ];
    calculateGlobalReturn(traders);

    // Main trace for the primary y-axis
  const trace1 = {
    x: roundSeries,
    y: priceSeries,
    mode: "lines+markers",
    type: "scatter",
    name: "Price",
    yaxis: "y2",
  };

  // Secondary trace for the secondary y-axis
  const trace2 = {
    x: roundSeries,
    y: secondarySeries,
    mode: "lines+markers",
    type: "scatter",
    name: "Return",
    yaxis: "y1", // Specify the second y-axis
  };
    let head = processName(trader["ResponseId"]) + " Performance";
      const layout = {
    title: head,
    xaxis: { title: "Rounds" },
    yaxis: {
      title: "Return",
      side: "left",
      range: [globalMinReturn, globalMaxReturn],
    },
    height: 300,
    yaxis2: {
      title: "Price",
      overlaying: "y",
      side: "right",
    },
    legend: {
      orientation: "h", // Horizontal legend
      x: 0.5, // Center it horizontally
      xanchor: "center",
      y: -0.4, // Position it below the plot
    },
  };
  const config = {
    displayModeBar: false, // this is the line that hides the bar.
  };
  Plotly.newPlot("eorchartdiv", [trace1, trace2], layout, config);
}

// Event listener for strategy buttons
document.querySelectorAll(".trader-button").forEach((button, index) => {
  const traderId = "player"+(index + 1);
  button.addEventListener("click", () => {
    const traderInfo = window[traderId];
	console.log(traderInfo);
    const traderResponseId = traderInfo["ResponseId"];
    const currentChartType = "strategy"; // Define the current chart type

    // Check if the same trader and chart type button was clicked
    if (
      lastTraderId === traderResponseId &&
      lastChartType === currentChartType
    ) {
      Plotly.purge("eorchartdiv"); // Clear the plot
      lastTraderId = null; // Reset the last trader
      lastChartType = null; // Reset the chart type
    } else {
      updateChart(traderInfo); // Update chart with new trader data
      lastTraderId = traderResponseId; // Update the last clicked trader
      lastChartType = currentChartType; // Update the last chart type
      roundData.plotClicks.push(traderResponseId);
      console.log(roundData.rankingClicks);
      console.log(roundData.plotClicks);
    }

    hideMessageRow();
  });
});

// Event listener for performance buttons
document.querySelectorAll(".trader-button-perf").forEach((button, index) => {
  const traderId = "player"+(index + 1);
  button.addEventListener("click", () => {
    const traderInfo = window[traderId];
    const traderResponseId = traderInfo["ResponseId"];
    const currentChartType = "performance"; // Define the current chart type

    // Check if the same trader and chart type button was clicked
    if (
      lastTraderId === traderResponseId &&
      lastChartType === currentChartType
    ) {
      Plotly.purge("eorchartdiv"); // Clear the plot
      lastTraderId = null; // Reset the last trader
      lastChartType = null; // Reset the chart type
    } else {
      updateChartPerf(traderInfo); // Update chart with new trader data
      lastTraderId = traderResponseId; // Update the last clicked trader
      lastChartType = currentChartType; // Update the last chart type
      roundData.rankingClicks.push(traderResponseId);
      console.log(roundData.rankingClicks);
      console.log(roundData.plotClicks);
    }

    hideMessageRow();
  });
});
/* taken out with implementation of buttons 22.11.
/// Info popup
const table = document.getElementById("copytable");
const rows = table.getElementsByTagName("tr");
const popup = document.getElementById("mypopup");
const popupName = document.getElementById("popupName");

for (let i = 2; i < rows.length; i++) {
  rows[i].addEventListener("click", function () {
    const targetCell = event.target;

    // Check if the clicked cell is in the last column
    if (
      targetCell.cellIndex === rows[i].cells.length - 1 ||
      targetCell.tagName === "BUTTON"
    ) {
      return; // Do nothing if it's the last column
    }

    event.stopPropagation();

    const cells = rows[i].getElementsByTagName("td");
    const name = cells[1].textContent;
    let thisPhaseReturn = "thisPhaseReturn" + i;
    let lastPhaseReturn = "lastPhaseReturn" + i;
    let allPhaseReturn = "allPhaseReturn" + i;
    let allPhaseGain = "allPhaseGain" + i;
    let roundseries = "roundseries" + i;
    let priceseries = "priceseries" + i;
    let returnseries = "returnseries" + i;
    let assetseries = "assetseries" + i;
    let rounds;
    let chart1 = "popupchartdiv2";
    let title1 = "Ongoing Returns";
    let righttitle1 = "Return in Percent";
    let chart2 = "popupchartdiv";
    let title2 = "The Number of Assets Held";
    let righttitle2 = "Number of Assets held";

    popupName.textContent = name;

    if (Stages[0].stage === "training" && roundData.round === 0) {
      rounds = Stages[0].rounds;
    } else if (Stages[0].stage === "training" && roundData.round === 4) {
      rounds = Stages[0].rounds;
    } else if (Stages[0].stage === "training" && roundData.round === 5) {
      rounds = Stages[1].rounds;
    } else {
      Stages[0].rounds;
    }

    PopupChartController(
      rounds,
      window[roundseries],
      window[returnseries],
      window[priceseries],
      chart1,
      title1,
      righttitle1,
      "line",
      0,
      name
    );
    /*
    PopupChartController(
      rounds,
      window[roundseries],
      window[assetseries],
      window[priceseries],
      chart2,
      title2,
      righttitle2,
      "bar",
      0
    );

    // Calculate the center of the screen
    const centerX = window.innerWidth / 2;
    const centerY = window.innerHeight / 2;

    // Calculate the positioning for the popup
    const popupWidth = popup.offsetWidth;
    const popupHeight = popup.offsetHeight;
    const popupLeft = centerX - popupWidth / 2;
    const popupTop = centerY - popupHeight / 2;

    // Set the popup position
    popup.style.left = popupLeft + "px";
    popup.style.top = popupTop + "px";
    popup.style.display = "block";
  });
}

// Add a click event listener to the popup to hide it when clicked inside
popup.addEventListener("click", function (event) {
  event.stopPropagation(); // Prevent the click event from propagating to the parent elements
  popup.style.display = "none"; // Close the popup when clicked inside
});

document.addEventListener("click", function (event) {
  if (!popup.contains(event.target)) {
    popup.style.display = "none";
  }
});

*/

/// Copying functionality
//Eventlistener for the copy buttons
var copyRank = null; // Initialize copyRank to null
// Function to handle button clicks
function handleButtonClick(event) {
  // Disallow clicking if phase is 9 and round is larger than 39
  if (phase === 9 && roundData.round > 39) {
    return; // Exit the function early
  }

  const buttonId = event.target.id;
  const buttonNumber = Number(buttonId.split("_")[2]) + page * 5;
  // Check if roundData.round is equal to 0
  // Toggle the .hidden class on other buttons
  for (let i = 1; i <= 5; i++) {
    const otherButton = document.getElementById("BUTTON_COPY_" + i);
    if (i != buttonNumber) {
      //otherButton.classList.add("hidden");
    }
  }

  // Toggle the .chosen class on the clicked button
  event.target.classList.toggle("chosen");
  event.target.textContent = "Copied";

  // If output is null, set it to the new value, otherwise, keep it as null
  if (copyRank === null) {
    copyRank = "rank_" + buttonNumber;
    // Store ID of copied trader
    var stage = "stage_" + phase;
    var round = "round_" + roundData.round;

    if (Stages[0].stage === "regular") {
      stage = "stage_" + (phase + 1);
      copiedTL = data[stage]["round_0"][copyRank]["ResponseId"];
      SCREENELEMENTS.decision_nextbutton.classList.remove("hidden");
      SCREENELEMENTS.copy_request.classList.add("hidden");
    } else if (Stages[0].stage === "training" && roundData.round > 0) {
      copiedTL = data[stage]["round_0"][copyRank]["ResponseId"];
      SCREENELEMENTS.decision_nextbutton.classList.remove("hidden");
      SCREENELEMENTS.copy_request.classList.add("hidden");
    } else {
      copiedTL = dataTraining[stage][round][copyRank]["ResponseId"];
      SCREENELEMENTS.decision_nextbutton_training.classList.remove("hidden");
      SCREENELEMENTS.copy_request.classList.add("hidden");
    }
    console.log(copiedTL);
    console.log(copyRank);
  } else {
    // If roundData.round is not 0, toggle the .hidden and .chosen classes
    for (let i = 1; i <= 5; i++) {
      const button = document.getElementById("BUTTON_COPY_" + i);
      button.classList.remove("hidden");
      button.classList.remove("chosen");
      button.textContent = "Copy";
      if (Stages[0].stage === "training") {
        SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
        SCREENELEMENTS.decision_nextbutton_training.classList.add("hidden");
        SCREENELEMENTS.copy_request.classList.remove("hidden");
      } else {
        SCREENELEMENTS.decision_nextbutton.classList.add("hidden");
        SCREENELEMENTS.decision_nextbutton_training.classList.add("hidden");
        SCREENELEMENTS.copy_request.classList.remove("hidden");
      }
    }

    copyRank = null; // Reset output to null
    copiedTL = null;

    //console.log("all buttons are available again.");

    //console.log(copyRank);
  }

  //console.log(copiedTL);
}

// Add event listeners to each button
for (let i = 1; i <= 5; i++) {
  const button = document.getElementById("BUTTON_COPY_" + i);
  button.addEventListener("click", handleButtonClick);
}

// Finding Rank dictionary based on Id saved when copy button is pressed
// Necessary because ranks change, to copy trading data of chosen trade leader
function copyMechanism(targetValue, phase, roundnr) {
  var foundKey = null;
  var stage = "stage_" + phase;
  var round = "round_" + roundData.round;

  if (Stages[0].stage === "training" && roundData.round < 5) {
    for (var key in dataTraining[stage][round]) {
      if (dataTraining[stage][round].hasOwnProperty(key)) {
        var innerDict = dataTraining[stage][round][key];
        for (var innerKey in innerDict) {
          if (
            innerDict.hasOwnProperty(innerKey) &&
            innerDict[innerKey] === targetValue
          ) {
            foundKey = key;
            break; // Exit the inner loop once a match is found
          }
        }
        if (foundKey !== null) {
          break; // Exit the outer loop if a match is found
        }
      }
    }
  } else if (Stages[0].stage === "training" && roundData.round === 5) {
    for (var key in data[stage]["round_0"]) {
      if (data[stage]["round_0"].hasOwnProperty(key)) {
        var innerDict = data[stage]["round_0"][key];
        for (var innerKey in innerDict) {
          if (
            innerDict.hasOwnProperty(innerKey) &&
            innerDict[innerKey] === targetValue
          ) {
            foundKey = key;
            break; // Exit the inner loop once a match is found
          }
        }
        if (foundKey !== null) {
          break; // Exit the outer loop if a match is found
        }
      }
    }
  } else {
    stage = "stage_" + phase;
    for (var key in data[stage][round]) {
      if (data[stage][round].hasOwnProperty(key)) {
        var innerDict = data[stage][round][key];
        for (var innerKey in innerDict) {
          if (
            innerDict.hasOwnProperty(innerKey) &&
            innerDict[innerKey] === targetValue
          ) {
            foundKey = key;
            break; // Exit the inner loop once a match is found
          }
        }
        if (foundKey !== null) {
          break; // Exit the outer loop if a match is found
        }
      }
    }
  }

  // Check if a matching key was found
  /*
  if (foundKey !== null) {
    console.log("Key for the target value:", foundKey);
  } else {
    console.log("Value not found in any keys.");
  }
*/

  if (copiedTL !== null) {
    if (Stages[0].stage === "training") {
      console.log(stage);
      console.log(round);
      console.log(copiedTL);
      console.log(foundKey);
      roundData.asset = dataTraining[stage][round][foundKey]["a"];
      roundData.cash = dataTraining[stage][round][foundKey]["c"];
      roundData.portfolio = dataTraining[stage][round][foundKey]["unrealized"];
    } else {
      roundData.asset = data[stage][round][foundKey]["a"];
      roundData.cash = data[stage][round][foundKey]["c"];
      roundData.portfolio = data[stage][round][foundKey]["unrealized"];
    }
  }
  return foundKey;
}

//Function that adjusts the copy buttons when browsing
function setButtonClasses(playerText) {
  // Get all the "player" cells in the table
  const playerCells = document.querySelectorAll(".player");
  //console.log(playerCells);

  playerCells.forEach((playerCell, index) => {
    // Get the text content of the player cell and remove any leading/trailing spaces
    const cellText = playerCell.textContent.trim();
    //console.log(cellText);
    //console.log(cellText);
    // Find the corresponding button by using the index (1-based) and constructing the class name
    const buttonClassName = ".BUTTON_COPY_" + (index + 1);
    //console.log(buttonClassName);
    // Find the button with the matching class name
    const button = document.querySelectorAll(buttonClassName);
    //console.log(button[0]);
    // Check if the button and player text exist
    if (treatment === 3) {
      if (button[0]) {
        if (playerText === null) {
          // If playerText is null, remove both classes
          button[0].classList.remove("chosen", "hidden");
          button[0].textContent = "Copy";
        } else if (cellText === playerText) {
          // If there's a match, add the "chosen" class and remove the "hidden" class
          button[0].classList.add("chosen");
          button[0].classList.remove("hidden");
          button[0].textContent = "Copied";
        } else if (cellText !== playerText) {
          // If there's no match, add the "hidden" class and remove the "chosen" class

          button[0].classList.add("hidden");
          button[0].classList.remove("chosen");
          button[0].textContent = "Copy";
        }
      }
    }
  });
}

function hideUnusedButtons(tableId) {
  const table = document.getElementById(tableId);

  for (let i = 1; i <= 5; i++) {
    const className = ".BUTTON_COPY_" + i;
    const buttons = table.querySelectorAll(className);
    //console.log(buttons);
    buttons.forEach(function (button, index) {
      //console.log(button);
      const row = button.closest("tr");
      const secondColumn = row.cells[1];

      // Check if the cell in the second column has a null value
      if (secondColumn.textContent.trim().toLowerCase() === "") {
        button.classList.add("hidden"); // Add the "hidden" class
      } else {
        if (treatment === 3) {
          button.classList.remove("hidden"); // Add the "hidden" class
        }
      }
    });
  }
}

function toggleColumnVisibility(show) {
  const columns = document.querySelectorAll(".toggle-column");
  columns.forEach((column) => {
    column.style.display = show ? "table-cell" : "none";
  });

  // Optionally hide the header
  const header = document.querySelector(".toggle-column-header");
  if (header) {
    header.style.display = show ? "table-cell" : "none";
  }
}
// Trade leader data

/*
/// Price path loading and selection function
const pp = {};
const pricePathsAll = {}

for (let j = 1; j < 51; j += 1) {
  let pathname = "path"+j;
  pricePathsAll[pathname] = [];
}

 let qualtricsPaths = QM.getEmbeddedData("pricepaths")
 var QPtemp = qualtricsPaths.split(';');			


for (let i = 0; i < 30; i += 1) {
  let propName =i+"period"; // create a dynamic property name
  pp[propName] = [];
  let temp = QPtemp[i]
  var temparr = temp.split(',');			
	//console.log(temparr);
  pp[propName].push(temparr);
  for (let j = 1; j < 51; j += 1) {
    let pathname = "path"+j;
    let temp = parseInt(pp[propName][0][j]);
      //console.log(pathname+" at period"+i+":"+pricePathsAll[pathname][i - 1]);
    if (pricePathsAll[pathname][i-1] === 0) {
      temp = 0;
    }
    pricePathsAll[pathname].push(temp);
  }
}


/// This function selects the price paths according to drawn priceseries (subtreatment)
const pathSeriesPicker = function (originalObject, pathseries) {
  const newObject = {};
  let startKey = 1+5*(pathseries-1)
  let endKey = 5+5*(pathseries-1)

     if (treatment == 1) {
    const pathName = "path" + (endKey + 1)
    const path0 = "path" + 0
    newObject[path0] = originalObject[pathName];
  };

  for (let key in originalObject) {
    const keyNumber = parseInt(key.match(/\d+/)[0]);

    if (keyNumber >= startKey && keyNumber <= endKey) {
      newObject[key] = originalObject[key];
    }
  }

  return newObject;
}

const pricePaths = pathSeriesPicker(pricePathsAll, pathseries)
console.log(pricePaths);
*/

/* TO DO:
- Treatment sorter
  - Price paths

  - Main treatment
- Copy mechanism
  - implementing results (copy from TL data)
  - forcing of the copying (cannot continue without)
  - Feedback screen after copying (why/how did you pick a specific trader)
- fewer phases and periods
*/

/*
function hideButtonsWithNullValueInSecondColumn(tableId) {
  const table = document.getElementById(tableId);

  for (let i = 1; i <= 5; i++) {
    const className = `BUTTON_COPY_${i}`;
    const buttons = table.querySelectorAll(`.${className}`);

    buttons.forEach(function (button, index) {
      const row = button.closest("tr");
      const secondColumn = row.cells[1];

      // Check if the cell in the second column has a null value
      if (secondColumn.textContent.trim().toLowerCase() === "null") {
        button.classList.add("hidden"); // Add the "hidden" class
      }
    });
  }
}

document.addEventListener("DOMContentLoaded", function () {
  hideButtonsWithNullValueInSecondColumn("myTable");
});
*/

var QM = new QualtricsManager();
initialize();
update();

const scrambledNameList = shuffleArray(nameList);
	pathseries = data["stage_0"]["round_0"]["rank_1"]["pathseries"];
console.log("pathseries test");
console.log(pathseries);
});


Qualtrics.SurveyEngine.addOnUnload(function () {
  /*Place your JavaScript here to run when the page is unloaded*/
});