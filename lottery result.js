function pickLotteryOutcome(x) {
  // Define the 5 possible lotteries as an array of objects
  const lotteries = [
    { y: 8000, z: 7200 },
    { y: 15000, z: 6300 },
    { y: 16000, z: 6050 },
    { y: 18600, z: 4700 },
    { y: 20000, z: 500 },
  ];

  // Choose one of the 5 lotteries based on the value of x
  // We assume x is between 0 and 4, as there are 5 lotteries
  const selectedLottery = lotteries[x % lotteries.length];

  // Randomly pick either y or z with equal chance
  const outcome = Math.random() < 0.5 ? selectedLottery.y : selectedLottery.z;
  // Output the value
  return outcome;
}

const x = 3; // Choose a value for x
const result = pickLotteryOutcome(x);
console.log("The result is:", result);
