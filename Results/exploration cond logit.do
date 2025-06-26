clear all
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
encode participantid, gen(id)

gen copiedcrra_str = string(copiedcrra)
encode copiedcrra_str, gen(alt)
list copiedcrra copiedcrra_str alt if _n <= 20

cmset id phase alt

gen rank_crra = rank * crra
gen rank_phase = rank * phase


cmxtmixlogit chosen return_bot returnall_bot, ///
    rand(rank_crra rank_phase) ///
    intpoints(10)
	
gen rank_phase_t1 = rank * phase * (combined_treatment == 0)
gen rank_phase_t2 = rank * phase * (combined_treatment == 1)
gen rank_phase_t3 = rank * phase * (combined_treatment == 2)
gen rank_phase_t4 = rank * phase * (combined_treatment == 3)

bysort id phase: egen var_rank = sd(rank)
list id phase rank var_rank if var_rank == 0


cmxtmixlogit chosen return_bot returnall_bot , ///
    rand( rank_phase_t3 rank_phase_t4) ///
    intpoints(10)

gen returnall_phase_t1 = returnall_bot * phase * (combined_treatment == 0)
gen returnall_phase_t2 = returnall_bot  * phase * (combined_treatment == 1)
gen returnall_phase_t3 = returnall_bot  * phase * (combined_treatment == 2)
gen returnall_phase_t4 = returnall_bot  * phase * (combined_treatment == 3)	
	
cmxtmixlogit chosen return_bot returnall_bot rank , ///
    rand(returnall_phase_t2 returnall_phase_t3 returnall_phase_t4) ///
    intpoints(10)

*risk level over time
gen riskrank_phase = copiedcrra_rank * phase
gen riskrank_phase_t1 = copiedcrra_rank * phase * (combined_treatment == 0)
gen riskrank_phase_t2 = copiedcrra_rank  * phase * (combined_treatment == 1)
gen riskrank_phase_t3 = copiedcrra_rank  * phase * (combined_treatment == 2)
gen riskrank_phase_t4 = copiedcrra_rank  * phase * (combined_treatment == 3)	
	
cmxtmixlogit chosen return_bot returnall_bot rank riskrank_phase, ///
    rand(riskrank_phase_t2 riskrank_phase_t3 riskrank_phase_t4) ///
	casevars(crra riskfalk) ///
    intpoints(50)


*risk attitude over time
gen crra_str = string(crra)
encode crra_str, gen(riskattitude)

gen riskatt_phase = riskattitude * phase
gen riskatt_phase_t1 = riskattitude * phase * (combined_treatment == 0)
gen riskatt_phase_t2 = riskattitude  * phase * (combined_treatment == 1)
gen riskatt_phase_t3 = riskattitude  * phase * (combined_treatment == 2)
gen riskatt_phase_t4 = riskattitude  * phase * (combined_treatment == 3)	
  
cmxtmixlogit chosen return_bot returnall_bot rank riskatt_phase riskatt_phase_t2 riskatt_phase_t3 riskatt_phase_t4, ///
	casevars(crra riskfalk) 

sort id phase


preserve
keep id phase gainall
duplicates drop

gen phase_lead = phase + 1  

rename gainall gainall_lead
drop phase
drop if phase_lead < 0
rename phase_lead phase
tempfile leadgain
save `leadgain'

restore


merge m:1 id phase using `leadgain', keepusing(gainall_lead) nogenerate


replace gainall_lead = 0 if phase == 0

list id phase gainall gainall_lead, sepby(phase)
drop if phase == 10

cmxtmixlogit chosen c.gainall_lead#c.return_bot, ///
	rand(return_bot returnall_bot rank) ///
	casevars(crra riskfalk gainall_lead riskatt_phase) ///
	intpoints(50)