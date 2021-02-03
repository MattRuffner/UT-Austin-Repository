---------------------------------------------
-- drop tables and sequences
---------------------------------------------
-- drop tables
drop table user_topic_subsc;
drop table video_topic_link;
drop table comments;
drop table topic;
drop table creditcard;
drop table video;
drop table content_creators;
drop table user_table;

-- drop sequences
DROP SEQUENCE card_id_seq;
DROP SEQUENCE cc_id_seq;
DROP SEQUENCE user_id_seq;
DROP SEQUENCE topic_id_seq;
DROP SEQUENCE video_id_seq;
DROP SEQUENCE comment_id_seq;

---------------------------------------------
-- create sequences
---------------------------------------------
-- Create card_id_seq
CREATE SEQUENCE card_id_seq
START WITH 100000 INCREMENT BY 1;

-- Create USER_id_seq
CREATE SEQUENCE user_id_seq
START WITH 10000 INCREMENT BY 1;

-- Create USER_id_seq
CREATE SEQUENCE cc_id_seq
START WITH 10000 INCREMENT BY 1;

-- Create topic_id_seq
CREATE SEQUENCE topic_id_seq
START WITH 1000 INCREMENT BY 1;

-- Create video_id_seq
CREATE SEQUENCE video_id_seq
START WITH 100000 INCREMENT BY 1;

-- Create comment_id_seq
CREATE SEQUENCE comment_id_seq
START WITH 1 INCREMENT BY 1;

---------------------------------------------
-- create tables
---------------------------------------------
CREATE TABLE user_table
(
    user_id         NUMBER    default user_id_seq.NEXTVAL    PRIMARY KEY,
    first_name      VARCHAR(50)     NOT NULL,
    middle_name     VARCHAR(50),
    last_name       VARCHAR(50)     NOT NULL,
    email           VARCHAR(50)     UNIQUE      NOT NULL,
    systemdate      DATE            DEFAULT SYSDATE,
    birthdate       DATE            NOT NULL,
    CC_flag         VARCHAR(1)      default 'N' NOT NULL,
    CONSTRAINT  age_over13_check    CHECK ((systemdate - birthdate)/365 > 13),
    CONSTRAINT  email_length_check  CHECK (LENGTH(email) >= 7)
    );

CREATE TABLE content_creators
(
    cc_id           NUMBER  default cc_id_seq.NEXTVAL     PRIMARY KEY,
    user_id         NUMBER,
    cc_username     VARCHAR(20)     UNIQUE,
    street_address  VARCHAR(30)     NOT NULL,
    city            VARCHAR(30)     NOT NULL,
    state           VARCHAR(30)     NOT NULL,
    zip_code        CHAR(5)         NOT NULL,
    state_residence VARCHAR(30)     NOT NULL,
    country_res     VARCHAR(30)     NOT NULL,
    mobile          CHAR(12)        NOT NULL,
    tier_level      VARCHAR(12)     DEFAULT 'Basic free',
    CONSTRAINT user_id_fk   FOREIGN KEY (user_id) REFERENCES user_table (user_id)
    );

CREATE TABLE creditcard
(
    card_id             NUMBER  default card_id_seq.NEXTVAL     PRIMARY KEY,
    contentcreator_id   NUMBER      REFERENCES content_creators (cc_id),
    card_type       VARCHAR(20)     NOT NULL,
    card_num        VARCHAR(16)     NOT NULL,
    CC_id           VARCHAR(4)      NOT NULL,
    street_billing  VARCHAR(30)     NOT NULL,
    city_billing    VARCHAR(30)     NOT NULL,
    state_billing   VARCHAR(30)     NOT NULL,
    zip_code_bill   CHAR(5)         NOT NULL
    );

CREATE TABLE video
(
    video_id        NUMBER  default video_id_seq.NEXTVAL    PRIMARY KEY,
    title           VARCHAR(60)     NOT NULL,
    subtitle        VARCHAR(120)    NOT NULL,
    upload_date     DATE            NOT NULL,
    video_length    NUMBER          NOT NULL,
    video_size      VARCHAR(10)     NOT NULL,
    cc_id           NUMBER          REFERENCES content_creators (cc_id),
    views           NUMBER  default '0'   NOT NULL,
    likes           NUMBER  default '0'   NOT NULL,
    revenue         NUMBER  default '0'   NOT NULL
    );

CREATE TABLE topic
(
    topic_id        NUMBER  default topic_id_seq.NEXTVAL    PRIMARY KEY,
    topic_name      VARCHAR(20)     NOT NULL,
    topic_desc      VARCHAR(60)     NOT NULL
    );

CREATE TABLE user_topic_subsc
(
    user_id         NUMBER  REFERENCES user_table (user_id),
    topic_id        NUMBER  REFERENCES topic (topic_id),
    CONSTRAINT user_topic_cpk   PRIMARY KEY (user_id, topic_id)
    );

CREATE TABLE video_topic_link
(
    video_id        NUMBER  REFERENCES video (video_id),
    topic_id        NUMBER  REFERENCES topic (topic_id),
    CONSTRAINT video_topic_cpk   PRIMARY KEY (video_id, topic_id)
    );

CREATE TABLE comments
(
    video_id        NUMBER  REFERENCES video (video_id),
    user_id         NUMBER  REFERENCES user_table (user_id),
    comment_id      NUMBER  default comment_id_seq.NEXTVAL PRIMARY KEY,
    time_date       DATE    NOT NULL,
    comment_body    VARCHAR(120)    NOT NULL
    );

---------------------------------------------
-- insert statements
---------------------------------------------
-- insert users
INSERT INTO user_table (first_name, middle_name, last_name, email, birthdate, CC_flag)
VALUES ('Clint','John','Tuttle','tuttle@mail.com','04-AUG-85','Y');

INSERT INTO user_table (first_name, middle_name, last_name, email, birthdate, CC_flag)
VALUES ('Tricia','Lynn','Moravec','tricia@mail.com','24-AUG-89','Y');

INSERT INTO user_table (first_name, last_name, email, birthdate, CC_flag)
VALUES ('Hamish','Cuthbert','ham@mail.com','04-JUL-87','N');

INSERT INTO user_table (first_name, last_name, email, birthdate, CC_flag)
VALUES ('Prabhudev','Garg','prabhu@mail.com','26-FEB-68','N');

INSERT INTO user_table (first_name, last_name, email, birthdate, CC_flag)
VALUES ('Lucy','Xiao','xiao@mail.com','01-JAN-03','Y');

INSERT INTO user_table (first_name, last_name, email, birthdate, CC_flag)
VALUES ('Tomoka','Kawahara','tomoka@mail.com','25-DEC-98','Y');

INSERT INTO user_table (first_name, last_name, email, birthdate, CC_flag)
VALUES ('Aditya','Yan','Aditya@mail.com','12-MAY-90','N');

INSERT INTO user_table (first_name, last_name, email, birthdate, CC_flag)
VALUES ('Lucy','Lui','lucylui@mail.com','02-DEC-68','Y');

INSERT INTO user_table (first_name, middle_name, last_name, email, birthdate, CC_flag)
VALUES ('Tej','S','Anand','anand@mail.com','16-NOV-63','Y');


-- Insert content creators
INSERT INTO content_creators (user_id, cc_username, street_address, city, state, zip_code, state_residence, country_res, mobile)
VALUES (10000,'CoolSQLSchool','123 Happy St','Austin','TX',78707,'Texas','US','015123437657');

INSERT INTO content_creators (user_id, cc_username, street_address, city, state, zip_code, state_residence, country_res, mobile, tier_level)
VALUES (10001,'LearnSQL','676 Burnet Rd','Austin','TX',78757,'Texas','US','018124339898','premium');

INSERT INTO content_creators (user_id, cc_username, street_address, city, state, zip_code, state_residence, country_res, mobile, tier_level)
VALUES (10001,'SQL Tips+Tricks','676 Burnet Rd','Austin','TX',78757,'Texas','US','018124339898','business');

INSERT INTO content_creators (user_id, cc_username, street_address, city, state, zip_code, state_residence, country_res, mobile)
VALUES (10004,'Hack Master','1414 Erie St','Chicago','IL',60654,'Illinois','US','012223335678');

INSERT INTO content_creators (user_id, cc_username, street_address, city, state, zip_code, state_residence, country_res, mobile)
VALUES (10005,'NoSQooL','1-7-7 Kabukicho','Shinjuku','CA',43051,'Tokyo','Japan','812228981144');

INSERT INTO content_creators (user_id, cc_username, street_address, city, state, zip_code, state_residence, country_res, mobile, tier_level)
VALUES (10007,'ElementarySQL','42 Stanford Ave','Brooklyn','NY',11209,'New York','US','premium','011234567890');

INSERT INTO content_creators (user_id, cc_username, street_address, city, state, zip_code, state_residence, country_res, mobile)
VALUES (10007,'Always practice!','42 Stanford Ave','Brooklyn','NY',11209,'New York','US','011234567890');

INSERT INTO content_creators (user_id, cc_username, street_address, city, state, zip_code, state_residence, country_res, mobile)
VALUES (10008,'Database Masterclass','123 Windsor Ave','Fairview','NY',13671,'New York','US','014189734782');

-- insert credit cards
INSERT INTO creditcard (contentcreator_id, card_type, card_num, CC_id, street_billing, city_billing, state_billing, zip_code_bill)
VALUES (10000, 'VISA','9910111222333444','123','123 Happy St','Austin','TX',78707);

INSERT INTO creditcard (contentcreator_id, card_type, card_num, CC_id, street_billing, city_billing, state_billing, zip_code_bill)
VALUES (10001, 'VISA','9910111222333444','123','676 Burnet Rd','Austin','TX',78757);

INSERT INTO creditcard (contentcreator_id, card_type, card_num, CC_id, street_billing, city_billing, state_billing, zip_code_bill)
VALUES (10001, 'VISA','9910111222333445','127','676 Burnet Rd','Austin','TX',78757);

INSERT INTO creditcard (contentcreator_id, card_type, card_num, CC_id, street_billing, city_billing, state_billing, zip_code_bill)
VALUES (10001, 'AMER','999911112222333','012','2222 Speedway','Austin','TX',78705);

INSERT INTO creditcard (contentcreator_id, card_type, card_num, CC_id, street_billing, city_billing, state_billing, zip_code_bill)
VALUES (10004, 'VISA','9251612345678956','989','1414 Erie St','Chicago','IL',60654);

INSERT INTO creditcard (contentcreator_id, card_type, card_num, CC_id, street_billing, city_billing, state_billing, zip_code_bill)
VALUES (10004, 'VISA','925161234567897','999','1414 Erie St','Chicago','IL',60654);

INSERT INTO creditcard (contentcreator_id, card_type, card_num, CC_id, street_billing, city_billing, state_billing, zip_code_bill)
VALUES (10005, 'VISA','9364758291048374','900','1-7-7 Kabukicho','Shinjuku','CA',43051);

INSERT INTO creditcard (contentcreator_id, card_type, card_num, CC_id, street_billing, city_billing, state_billing, zip_code_bill)
VALUES (10006, 'VISA','9361122334455667','100','42 Stanford Ave','Brooklyn','NY',11209);

-- Insert video
INSERT INTO video (title, subtitle, upload_date, video_length, video_size, cc_id, views, likes, revenue)
VALUES ('Only cool people use SQL', 'I use SQL to stay hip and cool ;)',TO_DATE('25-JAN-17 08:13:37','DD-MON-YY HH:MI:SS'),122,'2.4MB',10005,1098240,68700,78923);

INSERT INTO video (title, subtitle, upload_date, video_length, video_size, cc_id, views)
VALUES ('Database is the best', 'No really, its super fun',TO_DATE('12-AUG-20 12:02:07','DD-MON-YY HH:MI:SS'),1297,'2MB',10001,2103);

INSERT INTO video (title, subtitle, upload_date, video_length, video_size, cc_id, views, likes, revenue)
VALUES ('I will convince you', 'SQL is continuing to gain ground as a programming language',TO_DATE('13-AUG-20 12:14:07','DD-MON-YY HH:MI:SS'),456,'3MB',10001,25908,1240,6800);

INSERT INTO video (title, subtitle, upload_date, video_length, video_size, cc_id, views, likes, revenue)
VALUES ('PRO SQL Hacks', 'The best SQL hacks around',TO_DATE('07-MAY-19 12:58:21','DD-MON-YY HH:MI:SS'),365,'4.2MB',10004,10000000,34000,99987);

INSERT INTO video (title, subtitle, upload_date, video_length, video_size, cc_id, views, likes, revenue)
VALUES ('Elementary SQL', 'The Watson guide to SQL',TO_DATE('24-JUL-20 07:24:20','DD-MON-YY HH:MI:SS'),729,'4.7MB',10005,12345600,88000,120987);

INSERT INTO video (title, subtitle, upload_date, video_length, video_size, cc_id, views, likes, revenue)
VALUES ('Elementary SQL - part 2', 'The 2nd Watson guide to SQL',TO_DATE('25-JUL-20 07:25:20','DD-MON-YY HH:MI:SS'),803,'4.7MB',10005,18765600,97320,124356);

-- insert topics - OPTIONAL
INSERT INTO topic (topic_name,topic_desc)
VALUES ('learning','videos for learning concepts');

INSERT INTO topic (topic_name,topic_desc)
VALUES ('SQL','videos about SQL');

INSERT INTO topic (topic_name,topic_desc)
VALUES ('cool','videos about cool topics');

INSERT INTO topic (topic_name,topic_desc)
VALUES ('memes','images with memetic influence');

-- insert into user_topic_subsc
INSERT INTO user_topic_subsc (user_id, topic_id)
VALUES (10001,1000);

INSERT INTO user_topic_subsc (user_id, topic_id)
VALUES (10001,1001);

INSERT INTO user_topic_subsc (user_id, topic_id)
VALUES (10001,1002);

INSERT INTO user_topic_subsc (user_id, topic_id)
VALUES (10001,1003);

INSERT INTO user_topic_subsc (user_id, topic_id)
VALUES (10003,1001);

INSERT INTO user_topic_subsc (user_id, topic_id)
VALUES (10002,1001);

INSERT INTO user_topic_subsc (user_id, topic_id)
VALUES (10007,1001);

INSERT INTO user_topic_subsc (user_id, topic_id)
VALUES (10007,1002);

-- insert into video_topic_link
INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100000,1001);

INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100000,1002);

INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100003,1003);

INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100003,1002);

INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100002,1002);

INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100004,1001);

INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100004,1002);

INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100005,1001);

INSERT INTO video_topic_link (video_id, topic_id)
VALUES (100005,1002);

-- insert into comments
INSERT INTO comments (video_id, user_id, time_date, comment_body)
VALUES (100000,10002,TO_DATE('14-AUG-20 01:21:04','DD-MON-YY HH:MI:SS'),'Wow! What a not lame, super cool video!');

INSERT INTO comments (video_id, user_id, time_date, comment_body)
VALUES (100000,10003,TO_DATE('14-AUG-20 01:24:04','DD-MON-YY HH:MI:SS'),'Yeah, super cool and not lame at all. I don''t care what other say');

INSERT INTO comments (video_id, user_id, time_date, comment_body)
VALUES (100003,10000,TO_DATE('06-APR-19 11:45:09','DD-MON-YY HH:MI:SS'),'Wow! Amazing!');

INSERT INTO comments (video_id, user_id, time_date, comment_body)
VALUES (100003,10000,TO_DATE('06-APR-19 11:45:54','DD-MON-YY HH:MI:SS'),'But seriously! So cool!');

INSERT INTO comments (video_id, user_id, time_date, comment_body)
VALUES (100000,10004,TO_DATE('15-APR-19 10:15:23','DD-MON-YY HH:MI:SS'),'Great! So cool!');

INSERT INTO comments (video_id, user_id, time_date, comment_body)
VALUES (100000,10005,TO_DATE('06-APR-19 11:46:35','DD-MON-YY HH:MI:SS'),'Not bad');

INSERT INTO comments (video_id, user_id, time_date, comment_body)
VALUES (100000,10002,TO_DATE('06-APR-19 10:49:46','DD-MON-YY HH:MI:SS'),'Pretty neat!');

COMMIT;

-- indexes
CREATE INDEX creditcard_user_id_ix
ON creditcard (contentcreator_id);

CREATE INDEX video_cc_id_ix
ON video (cc_id);

CREATE INDEX comments_video_id_ix
ON comments (video_id);

CREATE INDEX comments_user_id_ix
ON comments (user_id);

-- start --

-- Matthew Ruffner
-- mar9632

-- Question 1
SELECT COUNT(DISTINCT cc.cc_id) AS total_number_of_content_creators,
MIN(v.video_length) AS shortest_video,
MAX(v.video_length) AS longest_video,
MAX(v.views) AS most_popular_video
FROM content_creators cc LEFT JOIN video v ON cc.cc_id = v.cc_id;


-- Question 2
SELECT COUNT(c.comment_id) AS number_of_comments,
MAX(c.time_date) AS most_recent_activity,
v.title AS video_title
FROM comments c JOIN video v ON c.video_id = v.video_id
GROUP BY v.title
ORDER BY MAX(c.time_date) ASC;

-- Question 3
SELECT COUNT(cc.cc_id) AS number_of_content_creators_in_city,
city,
ROUND(AVG(likes),2) AS avg_likes
FROM content_creators cc JOIN video v ON cc.cc_id = v.cc_id
GROUP BY city
ORDER BY AVG(likes) DESC;

-- Question 4
SELECT t.topic_id,
t.topic_name,
SUM(v.likes) AS tot_likes,
ROUND(AVG(TO_NUMBER(SUBSTR(v.video_size,1,length(v.video_size)-2))),2) AS avg_video_size
FROM topic t JOIN video_topic_link vt ON t.topic_id = vt.topic_id
    JOIN video v ON vt.video_id = v.video_id
GROUP BY t.topic_name, t.topic_id;

-- Question 4b
SELECT DISTINCT t.topic_id,
t.topic_name,
SUM(v.likes) OVER (PARTITION BY t.topic_name, t.topic_id) AS tot_likes,
ROUND(AVG(TO_NUMBER(SUBSTR(v.video_size,1,length(v.video_size)-2)))
    OVER (PARTITION BY t.topic_name, t.topic_id),2) AS avg_video_size
FROM topic t JOIN video_topic_link vt ON t.topic_id = vt.topic_id
    JOIN video v ON vt.video_id = v.video_id;

-- Question 5
SELECT u.first_name, u.last_name,
SUM(TRUNC((v.views - 100)/5000,0)) AS awards_earned
FROM user_table u INNER JOIN content_creators cc ON u.user_id = cc.user_id
    INNER JOIN video v ON v.cc_id = cc.cc_id
GROUP BY u.first_name, u.last_name
HAVING SUM(TRUNC((views - 100)/5000,0)) >= 10
ORDER BY SUM(TRUNC((v.views - 100)/5000,0)) DESC, u.last_name ASC;

-- Question 6a
SELECT COUNT(card_id) AS cards_per_person,
u.first_name,
cr.city_billing,
cr.state_billing
FROM user_table u  JOIN content_creators cc ON cc.user_id = u.user_id
     JOIN creditcard cr ON cc.cc_id = cr.contentcreator_id
WHERE cr.state_billing IN ('TX', 'NY')
GROUP BY u.first_name, ROLLUP(cr.city_billing, cr.state_billing)
ORDER BY cr.city_billing;

-- Question 6b
-- My understanding is that CUBE will do separate aggregations of values for all the columns in question
-- On the otherhand, ROLLUP does a hierarchical aggregation of values for the columns in question

-- Question 7
SELECT cc.street_address ||' '|| cc.city ||' '|| cc.state ||' '|| cc.zip_code AS cc_address,
cr.card_id,
CASE
WHEN cr.street_billing ||' '|| cr.city_billing ||' '|| cr.state_billing ||' '|| cr.zip_code_bill = cc.street_address ||' '|| cc.city ||' '|| cc.state ||' '|| cc.zip_code THEN 'Y'
ELSE 'N'
END AS flag
FROM content_creators cc JOIN creditcard cr ON cr.contentcreator_id = cc.cc_id;

-- Question 8
SELECT v.cc_id, COUNT(DISTINCT v.video_id) AS number_of_videos, COUNT(DISTINCT t.topic_name) AS number_of_topics
FROM video v JOIN video_topic_link vt ON v.video_id = vt.video_id
    JOIN topic t ON vt.topic_id = t.topic_id
GROUP BY v.cc_id
HAVING COUNT(DISTINCT t.topic_name) >= 2
ORDER BY cc_id DESC;


-- Question 8b
SELECT DISTINCT v.cc_id,
COUNT(DISTINCT v.video_id)
    OVER (PARTITION BY cc_id)  AS number_of_videos,
COUNT(DISTINCT t.topic_name)
    OVER (PARTITION BY cc_id) AS unique_topics
FROM video v JOIN video_topic_link vt ON v.video_id = vt.video_id
    JOIN topic t ON vt.topic_id = t.topic_id
ORDER BY cc_id DESC;

-- Question 9
SELECT DISTINCT topic_name
FROM topic t JOIN video_topic_link vtl
ON t.topic_id = vtl.topic_id JOIN video v
ON vtl.video_id = v.video_id
ORDER BY topic_name DESC;

-- Question 10
SELECT u.user_id, v.video_id, MAX(v.likes) AS most_liked_video
FROM video v JOIN content_creators cc ON v.cc_id = cc.cc_id
    JOIN user_table u ON cc.user_id = u.user_id
HAVING (MAX(likes) > (SELECT AVG(likes) FROM video))
GROUP BY (u.user_id, v.video_id)
ORDER BY MAX(likes) DESC;

-- Question 11
SELECT u.first_name, u.last_name, u.email, u.CC_flag, u.birthdate
FROM user_table u LEFT JOIN content_creators cc ON u.user_id = cc.user_id
    LEFT JOIN video v ON v.cc_id = cc.cc_id
WHERE u.user_id IN (SELECT u.user_id
    FROM user_table u LEFT JOIN content_creators cc ON u.user_id = cc.user_id
        LEFT JOIN video v ON v.cc_id = cc.cc_id
    HAVING COUNT(v.video_id) = 0
    GROUP BY u.user_id)
AND u.CC_flag = 'Y';

-- Question 12
SELECT *
FROM (SELECT v.title, v.subtitle, v.video_size, views, COUNT(c.comment_id) AS number_of_comments
    FROM video v LEFT JOIN comments c ON v.video_id = c.video_id
    GROUP BY (v.title, v.subtitle, v.video_size, views)
    HAVING COUNT(comment_id) >= 2
    ORDER BY COUNT(c.comment_id) DESC);

-- Question 13
SELECT u.user_id, u.first_name, u.last_name, COUNT(table1.video_id) AS number_of_videos
FROM user_table u LEFT JOIN (SELECT cc.user_id, v.video_id
        FROM content_creators cc  JOIN video v ON cc.cc_id = v.cc_id) table1
    ON u.user_id = table1.user_id
GROUP BY (u.user_id, u.first_name, u.last_name)
ORDER BY u.last_name DESC;

-- Question 14
SELECT cc_id, cc_username, ROUND((SYSDATE - recent_vid),2) AS days_since_latest_upload
FROM (SELECT MAX(v.upload_date) AS recent_vid, cc.cc_username, cc.cc_id
    FROM content_creators cc
        JOIN video v ON v.cc_id = cc.cc_id
    GROUP BY (cc.cc_id, cc.cc_username));
