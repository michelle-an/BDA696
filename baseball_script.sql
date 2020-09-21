USE baseball;
SHOW TABLES;


-- ################# HISTORIC BATTING AVERAGE #################
DROP TABLE IF EXISTS historic_batting_avg;

CREATE TABLE historic_batting_avg
	SELECT batter, SUM(Hit) as total_hits, SUM(atBat) as total_at_bats
	FROM baseball.batter_counts bc
	GROUP BY batter;

ALTER TABLE historic_batting_avg ADD average DECIMAL (4, 3);

UPDATE historic_batting_avg SET average = total_hits/total_at_bats WHERE total_at_bats > 0;
UPDATE historic_batting_avg SET average = 0.000 WHERE total_at_bats = 0;

SELECT * FROM historic_batting_avg;


-- ################# ANNUAL BATTING AVERAGE #################
SET @year_end_date = DATE('2008-12-31'); -- use this variable to set the last day to include in the annual search window

DROP TABLE IF EXISTS annual_avg_temp;
DROP TABLE IF EXISTS annual_batting_avg;

CREATE TABLE annual_avg_temp
	SELECT bc.batter, bc.Hit, bc.atBat, gt.local_date
	FROM batter_counts bc
	INNER JOIN game_temp gt ON bc.game_id = gt.game_id
	WHERE gt.local_date BETWEEN DATE_ADD(@year_end_date, INTERVAL -1 YEAR) AND @year_end_date;

SELECT * FROM annual_avg_temp ORDER BY batter;

CREATE TABLE annual_batting_avg
	SELECT batter, SUM(Hit) as total_hits, SUM(atBat) as total_at_bats
	FROM annual_avg_temp
	GROUP BY batter;

ALTER TABLE annual_batting_avg ADD annual_batting_average decimal (4,3);

UPDATE annual_batting_avg SET annual_batting_average = FORMAT((total_hits/total_at_bats), 3) WHERE total_at_bats > 0;
UPDATE annual_batting_avg SET annual_batting_average = 0 WHERE total_at_bats = 0;

SELECT * FROM annual_batting_avg;


-- ################# ROLLING BATTING AVERAGE #################
SET @end_date = DATE('2011-06-30'); -- use this variable to set the last day to include in the 100 day search window

DROP TABLE IF EXISTS rolling_avg_temp;
DROP TABLE IF EXISTS rolling_batting_avg;

CREATE TABLE rolling_avg_temp
	SELECT bc.batter , bc.Hit, bc.atBat
	FROM batter_counts bc
	INNER JOIN game_temp gt ON bc.game_id = gt.game_id
	WHERE local_date BETWEEN DATE_ADD(@year_end_date, INTERVAL -100 DAY) AND @year_end_date;

SELECT * FROM rolling_avg_temp;

CREATE TABLE rolling_batting_avg
	SELECT batter, sum(Hit) AS total_hits, sum(atBat) AS total_at_bats
	FROM rolling_avg_temp
	GROUP BY batter;

ALTER TABLE rolling_batting_avg ADD rolling_batting_average decimal (4,3);

UPDATE rolling_batting_avg SET rolling_batting_average = FORMAT((total_hits/total_at_bats), 3) WHERE total_at_bats > 0;
UPDATE rolling_batting_avg SET rolling_batting_average = 0 WHERE total_at_bats = 0;

SELECT * FROM rolling_batting_avg;
