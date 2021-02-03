-- Matthew Ruffner mar9632

-- Dropping tables before we start
DROP TABLE Topics_Videos;
DROP TABLE Comments_Videos;
DROP TABLE User_Topic;
DROP TABLE CClinkedID_videos;
DROP TABLE Topics;
DROP TABLE Videos;
DROP TABLE Comments;
DROP TABLE Content_Creators;
DROP TABLE Payments;
DROP TABLE Users;






COMMIT;

-- Dropping sequences before we create them
DROP SEQUENCE Video_ID_seq;
DROP SEQUENCE Comment_ID_seq;
DROP SEQUENCE User_ID_seq;
DROP SEQUENCE CC_ID_seq;
DROP SEQUENCE Card_ID_seq;
DROP SEQUENCE Topic_ID_seq;

COMMIT;

-- creating sequences that will end up being used for creating unique primary keys
CREATE SEQUENCE Video_ID_seq
START WITH 1000000000 INCREMENT BY 1
MINVALUE 1000000000 MAXVALUE 9999999999
NOCYCLE;

CREATE SEQUENCE Comment_ID_seq
START WITH 1000000000 INCREMENT BY 1
MINVALUE 1000000000 MAXVALUE 9999999999
NOCYCLE;

CREATE SEQUENCE User_ID_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999
NOCYCLE;

CREATE SEQUENCE CC_ID_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999
NOCYCLE;

CREATE SEQUENCE Card_ID_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999
NOCYCLE;

CREATE SEQUENCE Topic_ID_seq
START WITH 1000000 INCREMENT BY 1
MINVALUE 1000000 MAXVALUE 9999999
NOCYCLE;

COMMIT;

-- creating our tables and entities to put our data into later with required constraints if necessary
CREATE TABLE Users (
  User_ID NUMBER(7) DEFAULT User_ID_seq.nextval PRIMARY KEY,
  First_Name VARCHAR(20) NOT NULL,
  Middle_Name VARCHAR(20),
  Last_Name VARCHAR(20) NOT NULL,
  Email_Address VARCHAR(40) UNIQUE NOT NULL
    CONSTRAINT email_length_check CHECK (LENGTH(Email_Address) >= 7),
  Birthdate DATE NOT NULL
    CONSTRAINT age_check_constraint CHECK (DATE '2020-09-20' - Birthdate >= 4745),
  CC_Flag CHAR(1) DEFAULT 'N' NOT NULL
);

CREATE TABLE Payments (
  Card_ID NUMBER(7) DEFAULT Card_ID_seq.nextval PRIMARY KEY,
  User_ID NUMBER(7),
  Card_Number NUMBER(16) NOT NULL,
  Type_of_Card VARCHAR(20) NOT NULL,
  Expiration_Date DATE NOT NULL,
  Security_Code NUMBER(3) NOT NULL,
  Billing_City VARCHAR(20) NOT NULL,
  Billing_State CHAR(2) NOT NULL,
  Billing_Zipcode NUMBER(5) NOT NULL,
  CONSTRAINT FK_user_id_pay FOREIGN KEY (User_ID)
    REFERENCES Users(User_ID)
);

CREATE TABLE Content_Creators (
  CC_ID NUMBER(7) DEFAULT CC_ID_seq.nextval PRIMARY KEY,
  User_ID NUMBER(7),
  Username VARCHAR(30) NOT NULL,
  Country CHAR(20) NOT NULL,
  State_of_Residence CHAR(2) NOT NULL,
  Mobile_Phone CHAR(12) NOT NULL,
  Subscriptions VARCHAR(15) NOT NULL,
  CONSTRAINT FK_user_id_cc FOREIGN KEY (User_ID)
    REFERENCES Users(User_ID)
);

CREATE TABLE Comments (
  Comment_ID NUMBER(9) DEFAULT Comment_ID_seq.nextval PRIMARY KEY,
  Comment_Description VARCHAR(250) NOT NULL,
  Comment_Time DATE NOT NULL,
  User_ID NUMBER(7),
  CONSTRAINT FK_user_id_com FOREIGN KEY (User_ID)
    REFERENCES Users(User_ID)
);

CREATE TABLE Videos (
  Video_ID NUMBER(9) DEFAULT Video_ID_seq.nextval PRIMARY KEY,
  CC_ID NUMBER(7),
  Video_Title VARCHAR(100) NOT NULL,
  Video_Subtitle VARCHAR(100) NOT NULL,
  Date_Uploaded DATE NOT NULL,
  Video_Length NUMBER(5,2) NOT NULL,
  Views NUMBER(9) DEFAULT 0,
  Likes NUMBER(7) DEFAULT 0,
  Revenue NUMBER(7) DEFAULT 0,
  Video_Size NUMBER(6) NOT NULL,
  CONSTRAINT FK_cc_id_vid FOREIGN KEY (CC_ID)
    REFERENCES Content_Creators(CC_ID)
);


CREATE TABLE Topics (
  Topic_ID NUMBER(7) DEFAULT Topic_ID_seq.nextval PRIMARY KEY,
  Topic_Name VARCHAR(30) NOT NULL,
  Topic_Description VARCHAR(150) NOT NULL
);

CREATE TABLE Topics_Videos (
  Topic_ID NUMBER(7) NOT NULL,
  Video_ID NUMBER(9) NOT NULL,
  PRIMARY KEY (Topic_ID, Video_ID),
  CONSTRAINT FK_topic_id_tv FOREIGN KEY (Topic_ID)
    REFERENCES Topics(Topic_ID),
  CONSTRAINT FK_video_id_tv FOREIGN KEY (Video_ID)
    REFERENCES Videos(Video_ID)
);


CREATE TABLE Comments_Videos (
  Video_ID NUMBER(9),
  Comment_ID NUMBER(9),
  PRIMARY KEY (Video_ID, Comment_ID),
  CONSTRAINT FK_comment_id_cv FOREIGN KEY (Comment_ID)
    REFERENCES Comments(Comment_ID),
  CONSTRAINT FK_video_id_cv FOREIGN KEY (Video_ID)
    REFERENCES Videos(Video_ID)
);

CREATE TABLE User_Topic (
  Topic_ID NUMBER(7),
  User_ID NUMBER(7),
  PRIMARY KEY (Topic_ID, User_ID),
  CONSTRAINT FK_user_id_ut FOREIGN KEY (User_ID)
    REFERENCES Users(User_ID),
  CONSTRAINT FK_topic_id_ut FOREIGN KEY (Topic_ID)
    REFERENCES Topics(Topic_ID)
);


CREATE TABLE CClinkedID_videos (
  CC_ID NUMBER(7),
  Video_ID NUMBER(9),
  PRIMARY KEY (CC_ID, Video_ID),
  CONSTRAINT FK_CC_id_ccv FOREIGN KEY (CC_ID)
    REFERENCES Content_Creators(CC_ID),
  CONSTRAINT FK_video_id_ccv FOREIGN KEY (Video_ID)
    REFERENCES Videos(Video_ID)
);

COMMIT;

-- putting data into our tables
INSERT INTO Users (First_Name, Middle_Name, Last_Name, Email_Address, Birthdate, CC_Flag)
Values ('Matthew', 'Alessi', 'Ruffner', 'mruffner@gmail.com', TO_DATE('1997-02-03', 'YYYY-MM-DD'), 'Y');
COMMIT;

--SELECT * FROM Users;

INSERT INTO Users (First_Name, Middle_Name, Last_Name, Email_Address, Birthdate, CC_Flag)
Values ('Benjamin', 'Perea', 'Deutsch', 'bdeutsch@gmail.com', TO_DATE('1992-12-23', 'YYYY-MM-DD'), 'Y');
COMMIT;

INSERT INTO Users (First_Name, Last_Name, Email_Address, Birthdate, CC_Flag)
Values ('Yiqun', 'Tian', 'yitian@gmail.com', TO_DATE('1996-08-13', 'YYYY-MM-DD'), 'Y');
COMMIT;

INSERT INTO Users (First_Name, Last_Name, Email_Address, Birthdate, CC_Flag)
Values ('Sidd', 'Chauhan', 'schauhan@gmail.com', TO_DATE('1997-02-03', 'YYYY-MM-DD'), 'Y');
COMMIT;

INSERT INTO Users (First_Name, Last_Name, Email_Address, Birthdate)
Values ('Sunny', 'Vidhani', 'svidhani@gmail.com', TO_DATE('1997-04-29', 'YYYY-MM-DD'));
COMMIT;

INSERT INTO Users (First_Name, Last_Name, Email_Address, Birthdate)
Values ('Luke', 'Stevens', 'lstevens@gmail.com', TO_DATE('1997-11-05', 'YYYY-MM-DD'));
COMMIT;

--SELECT * FROM Content_Creators;

INSERT INTO Content_Creators (User_ID, Subscriptions, Username, Country, State_of_Residence, Mobile_Phone)
Values (1000000, 'Free', 'mattruffner', 'US', 'TX','010002382838');
COMMIT;

INSERT INTO Content_Creators (User_ID, Subscriptions, Username, Country, State_of_Residence, Mobile_Phone)
Values (1000001, 'Business', 'bendutch', 'US', 'AK','010002382228');
COMMIT;

INSERT INTO Content_Creators (User_ID, Subscriptions, Username, Country, State_of_Residence, Mobile_Phone)
Values (1000002, 'General', 'sidishere', 'US', 'VA','010005682838');
COMMIT;

INSERT INTO Content_Creators (User_ID, Subscriptions, Username, Country, State_of_Residence, Mobile_Phone)
Values (1000003, 'Free', 'sunnyday', 'US', 'TX','010002382333');
COMMIT;

--SELECT * FROM Payments;

INSERT INTO Payments (User_ID, Card_Number, Type_of_Card, Expiration_Date, Security_Code, Billing_City, Billing_State, Billing_Zipcode)
Values (1000000, 1234000012340000,'Visa', TO_DATE('2021-11-05', 'YYYY-MM-DD'), 123, 'Mclean', 'TX','22102');
COMMIT;

INSERT INTO Payments (User_ID, Card_Number, Type_of_Card, Expiration_Date, Security_Code, Billing_City, Billing_State, Billing_Zipcode)
Values (1000000, 1234000012340001,'MasterCard', TO_DATE('2022-11-05', 'YYYY-MM-DD'), 323, 'Mclean', 'TX','22102');
COMMIT;

INSERT INTO Payments (User_ID, Card_Number, Type_of_Card, Expiration_Date, Security_Code, Billing_City, Billing_State, Billing_Zipcode)
Values (1000001, 4321000012340000,'Visa', TO_DATE('2021-02-15', 'YYYY-MM-DD'), 222, 'Anchorage', 'AK','42873');
COMMIT;

INSERT INTO Payments (User_ID, Card_Number, Type_of_Card, Expiration_Date, Security_Code, Billing_City, Billing_State, Billing_Zipcode)
Values (1000002, 1234111112340000,'MasterCard', TO_DATE('2023-11-23', 'YYYY-MM-DD'), 987, 'Arlington', 'VA','34256');
COMMIT;

--SELECT * FROM Videos;

ALTER SESSION SET nls_date_format = 'mm/dd/yy HH24:MI:SS';

-- there was a weird error for this section and the next that I wasn't able to diagnose. The code should be good but it isn't running unfortunately
INSERT INTO Videos (CC_ID, Video_Title, Video_Subtitle, Date_Uploaded, Video_Length, Views, Likes, Revenue, Video_Size)
Values (1000000, 'Yi eats apple', 'yiapple', '11/23/19 10:37:22', '34.22', '5', '2', '1', '22');
COMMIT;

INSERT INTO Videos (CC_ID, Video_Title, Video_Subtitle, Date_Uploaded, Video_Length, Video_Size)
Values (1000001, 'MJ vs. Lebron', 'mjbron', '09/20/20 10:58:22', '234.12', '223');
COMMIT;

INSERT INTO Videos (CC_ID, Video_Title, Video_Subtitle, Date_Uploaded, Video_Length, Views, Likes, Revenue, Video_Size)
Values (1000002, 'pickmin', 'game', '08/18/18 03:20:58', '1.45', '11', '5', '3', '21');
COMMIT;

INSERT INTO Videos (CC_ID, Video_Title, Video_Subtitle, Date_Uploaded, Video_Length, Views, Likes, Revenue, Video_Size)
Values (1000003, 'Yi eats apple', 'yiapple', '11/23/19 10:37:22', '29.39', '20000', '222', '18', '7638');
COMMIT;

INSERT INTO Videos (CC_ID, Video_Title, Video_Subtitle, Date_Uploaded, Video_Length, Views, Likes, Revenue, Video_Size)
Values (1000003, 'Yi eats apple', 'yiapple', '11/23/19 10:37:22', '29.39', '20000', '222', '18', '7638');
COMMIT;

--SELECT * FROM Comments;

INSERT INTO Comments (Comment_Description, Comment_Time, User_ID)
Values ('this was bad lol', '11/23/19 10:37:22', 1000001);
COMMIT;

INSERT INTO Comments (Comment_Description, Comment_Time, User_ID)
Values ('interesting', '06/18/18 11:23:32', 1000002);
COMMIT;

INSERT INTO Comments (Comment_Description, Comment_Time, User_ID)
Values ('hey this is cool', '04/11/19 07:11:25', 1000003);
COMMIT;

INSERT INTO Comments (Comment_Description, Comment_Time, User_ID)
Values ('why did this happen', '11/24/19 10:47:22', 1000004);
COMMIT;

-- Creating indexes for foreign keys
CREATE INDEX FK_user_id_pay_index ON Payments (User_ID);
CREATE INDEX FK_user_id_cc_index ON Content_Creators (User_ID);
CREATE INDEX FK_user_id_com_index ON Comments (User_ID);
CREATE INDEX FK_cc_id_com_index ON Videos (CC_ID);
