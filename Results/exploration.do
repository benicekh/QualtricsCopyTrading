clear all

import delimited "C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Results\Pilot\Cleaned\panel_data_cleaned_bot.csv"

encode participantid, gen(id)

xtset id phase

gen combined_treatment = .
replace combined_treatment = 0 if nametreatment == 0 & followerstreatment == 0
replace combined_treatment = 1 if nametreatment == 1 & followerstreatment == 0
replace combined_treatment = 2 if nametreatment == 0 & followerstreatment == 1
replace combined_treatment = 3 if nametreatment == 1 & followerstreatment == 1

label define treatment_lbl ///
    0 "No Name, No Followers" ///
    1 "Name Only" ///
    2 "Followers Only" ///
    3 "Name and Followers"

label values combined_treatment treatment_lbl

encode tradeleader, gen(TL)
codebook TL

mlogit TL i.copiedrank##i.combined_treatment i.overallrank##i.combined_treatment c.phase##i.combined_treatment c.return_bot##i.combined_treatment c.returnall_bot##i.combined_treatment i.crra_rank##i.combined_treatment, baseoutcome(2)

margins, predict(pr) at(combined_treatment==(0(1)3)) level(90)
marginsplot , title("")


margins, predict(pr) at(combined_treatment==(0(1)3) copiedrank==(1(1)5) ) level(90)
marginsplot , title("")

margins, predict(pr) at( crra_rank==(1(1)5) combined_treatment==(0(1)3)) level(90)
marginsplot , title("")

margins, predict(pr) at(phase==(0(1)9) combined_treatment==(0(1)3)) level(90)
marginsplot , title("")

margins, predict(pr) at(overallrank==(1(1)5) combined_treatment==(0(1)3)) level(90)
marginsplot , title("")

margins, predict(pr) at(phase==(0(1)9) copiedrank==(1(1)5)) level(90)
marginsplot , title("")

summarize return_bot
margins, at(return_bot=(-35(5)50) combined_treatment==(0(1)3)) predict(outcome(1))
marginsplot , title("")

margins, at(return_bot=(-20(5)30) combined_treatment==(0(1)3)) predict(outcome(5))
marginsplot , title("")

summarize returnall_bot
margins, at(returnall_bot=(-20(5)30) combined_treatment==(0(1)3)) predict(outcome(1))
marginsplot , title("")

margins, at(returnall_bot=(-20(5)30) combined_treatment==(0(1)3)) predict(outcome(2))
marginsplot , title("")

margins, at(returnall_bot=(-20(5)30) combined_treatment==(0(1)3)) predict(outcome(3))
marginsplot , title("")

margins, at(returnall_bot=(-20(5)30) combined_treatment==(0(1)3)) predict(outcome(4))
marginsplot , title("")

margins, at(returnall_bot=(-20(5)30) combined_treatment==(0(1)3)) predict(outcome(5))
marginsplot , title("")

margins, at(copiedrank=(1(1)5) combined_treatment==(0(1)3)) predict(outcome(1))
marginsplot , title("")

marginsplot, xdimension(copiedrank) ///
    title("Predicted Probability of TL = 1") ///
    ylabel(, angle(horizontal)) ///
    xlabel(, grid) ///
    legend(title("combined_treatment"))

preserve
keep if pathseries==2	
mlogit TL i.phase##i.combined_treatment, baseoutcome(2)
margins, at(phase=(0(1)9) combined_treatment==(0(1)3)) predict(outcome(1))
marginsplot , title("")

margins, at(phase=(0(1)9) combined_treatment==(0(1)3)) predict(outcome(5))
marginsplot , title("")
restore

import delimited "C:\Users\U338144\surfdrive - Hiepler, B. (Benjamin)@surfdrive.surf.nl\Projects\Copy Trading\Trading game\Results\Pilot\Cleaned\cmclogit_panel.csv", clear
egen group_id = group(participantid phase)
gen combined_treatment = .
replace combined_treatment = 0 if nametreatment == 0 & followerstreatment == 0
replace combined_treatment = 1 if nametreatment == 1 & followerstreatment == 0
replace combined_treatment = 2 if nametreatment == 0 & followerstreatment == 1
replace combined_treatment = 3 if nametreatment == 1 & followerstreatment == 1

label define treatment_lbl ///
    0 "No Name, No Followers" ///
    1 "Name Only" ///
    2 "Followers Only" ///
    3 "Name and Followers"

label values combined_treatment treatment_lbl

cmset group_id tradeleader

cmclogit chosen return_bot, ///
    basealternative(CRRA_0)
	
	
	
	
	
	
	
xtreg overwrittenlottery i.combined_treatment, robust
reg overwrittenlottery i.combined_treatment if phase == 0, robust
reg overwrittenlottery i.combined_treatment if phase == 9, robust
xtreg copiedrank i.combined_treatment, robust
xtreg copiedrank i.combined_treatment##c.phase##i.pathseries, robust
reg copiedrank i.combined_treatment if phase == 0, robust
reg copiedrank i.combined_treatment if phase == 9, robust

reg copiedrank i.combined_treatment i.phase##i.pathseries, robust
reg overwrittenlottery i.combined_treatment i.phase##i.pathseries, robust
reg gain i.combined_treatment i.phase##i.pathseries, robust

xtreg gain i.combined_treatment#i.phase, robust

