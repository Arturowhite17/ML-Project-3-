-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- Link to schema: https://app.quickdatabasediagrams.com/#/d/iDTteS
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.

-- Modify this code to update the DB schema diagram.
-- To reset the sample schema, replace everything with
-- two dots (".." - without quotes).
-- * Create a conceptual ERD by determining the entities that will be present in the database, along with their attributes.
-- Be sure to include the following: `trainers`, `members`, and `gym` as well as one more entity that you think is necessary.


CREATE TABLE "hud_dataset" (
	"GEOID" int NOT NULL,
	"Subsidized units available" int NOT NULL,
	"% Occupied" dec NOT NULL,
	"% Reported" dec NOT NULL,
    "Household income per year" int NOT NULL,
    "% very low income" dec NOT NULL,
    "% extremely low income" dec NOT NULL, 
    "% female head" dec NOT NULL,
    "% with disability among Head Spouse Co-head aged 61 years or less" dec NOT NULL,
    "% with disability among Head Spouse Co-head aged 62 years or older" dec NOT NULL,
    "% in poverty (Census tract)", "% minority (Census tract)" dec NOT NULL,
    FOREIGN KEY (GEOID) REFERENCES censustract(GEOID)
);

CREATE TABLE "censustract" (
	"GEOID" int(11) NOT NULL,
	"StateID" int(2) NOT NULL,
	"CountyID" int(3) NOT NULL,
	"TractID" int(6) NOT NULL,
	"TractName" char(20),
	"Latitude" dec NOT NULL,
	"Longitude" dec NOT NULL,
    CONSTRAINT "pk_censustract" PRIMARY KEY (
        "GEOID"
     )
);

CREATE TABLE "census_dataset" (
	"GEOID" int NOT NULL,
	"RentBurdenedHHs" int NOT NULL,
	"TotalLowIncomeHHs" int NOT NULL,
	"MedianIncomeHH" int NOT NULL,
    FOREIGN KEY (GEOID) REFERENCES censustract(GEOID)
);

CREATE TABLE "nhpd_dataset" (
	"GEOID" int NOT NULL,
	"NHPD Property ID" int(8) NOT NULL,
	"PropertyStatus" char(10),
	"Property Name" char(30),
	"Property Address" char(100),
	"City" char(20),
	"State" char(2),
	"Zip" char(15),
	"Active Subsidies" int,
	"TargetTenantType" char(20),
	"Occupancy Rate" int,
	"0-1 BedroomUnits" int,
	"TwoBedroomUnits" int,
	"ThreePlusBedroomUnits" int,
	"Percent of ELI Households" int,
	"HUD Property ID" char(30),
	"FairMarketRent_2BR" int,
	"Owner Type" char(10),
	"EarliestStartDate" date,
	"Total Units" int(4),
	"Section 8 Housing Units" int(4),
	"Supportive Housing Elderly" int(4),
	"FHA Loan Units" int(4),
	"FHA Loan Sub_Program Units" int(4),
	"Low Income Housing Tax Credit" int(4),
	"Low Income Housing Tax Credit sub-program" int(4),
	"HOME Grants" int(4),
	"Rural Rental Housing" int(4),
    FOREIGN KEY (GEOID) REFERENCES censustract(GEOID)
);


