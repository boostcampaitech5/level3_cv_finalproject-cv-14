const express = require("express");
const app = express();
const session = require("express-session");
const cors = require("cors");
const FormData = require("form-data");
const mongoose = require("mongoose");
const MongoDBStore = require("connect-mongodb-session")(session);
const passportLocalMongoose = require("passport-local-mongoose");
const passport = require("passport");
const aws = require("aws-sdk");
const dotenv = require("dotenv");
const multer = require("multer");
const multerS3 = require("multer-s3");
const path = require("path");
const uuid4 = require("uuid4");
const axios = require("axios");

require("aws-sdk/lib/maintenance_mode_message").suppress = true;

dotenv.config();

app.use(express.static("public"));
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// DB연결
mongoose.connect(process.env.DB_URL, {
  // useUnifiedTopology: true,
  // useNewUrlParser: true
});

const db = mongoose.connection;

const handleOpen = () => console.log("Connected to DB");
const handleError = (error) => console.log(`error on DB connection:${error}`);

db.once("open", handleOpen);
db.on("error", handleError);

// DB session 저장
var store = new MongoDBStore({
  uri: process.env.DB_URL,
  collection: "mySessions",
});

// Catch errors
store.on("error", function (error) {
  console.log(error);
});

app.use(
  session({
    secret: "dwdasjfu123@#@fc",
    resave: true,
    saveUninitialized: false,
    store: store,
  })
);
app.use(passport.initialize());
app.use(passport.session());
app.set("view engine", "ejs");

// aws region 및 자격증명 설정
const s3 = new aws.S3({
  region: process.env.AWS_S3_REGION,
  accessKeyId: process.env.AWS_S3_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_S3_SECRET_ACCESS_KEY,
});

//회원가입 스키마 지정
const UserSchema = new mongoose.Schema({
  name: String,
  email: String,
  // 사용자마다 이미지 정보를 저장하는 필드 추가
  imagePaths: [
    {
      imagePath: String,
      embeddingVector: [Number], // 임베딩 벡터는 숫자 배열로 저장합니다.
      uploadTime: { type: Date, default: Date.now }, // 파일 업로드 시간 정보를 저장하는 필드
    },
  ],
});
// 이메일을 기준으로 자동으로 검사
UserSchema.plugin(passportLocalMongoose, { usernameField: "email" });

const User = mongoose.model("User", UserSchema);

passport.use(User.createStrategy());

passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

// var db;
// MongoClient.connect(process.env.DB_URL, function(err, client) {
//     // 연결되면 할일
//     if (err) return console.log(err);

//     db = client.db("final-project");

//     // db.collection("post").insertOne({ 이름: "John", 나이: 20 }, function (에러, 결과) {
//     //   console.log("저장완료");
//     // });

// });

// 확장자 검사 목록
const allowedExtensions = [".png", ".jpg", ".jpeg"];

const uploadImage = multer({
  storage: multerS3({
    s3: s3,
    bucket: process.env.AWS_S3_BUCKET_NAME,
    contentType: multerS3.AUTO_CONTENT_TYPE,
    key: (req, file, callback) => {
      // 콜백 함수 두 번째 인자에 파일명(경로 포함)을 입력
      //const uploadDirectory = req.query.directory ?? ''
      // 확장자 검사
      const extension = path.extname(file.originalname).toLowerCase();
      if (!allowedExtensions.includes(extension)) {
        return callback(new Error("wrong extension"));
      }
      const userFolder = req.user.email;
      const uniqueFileName = `${userFolder}/${Date.now()}_${uuid4()}${extension}`;
      callback(null, uniqueFileName);
    },
    acl: "public-read-write",
  }),
  // 이미지 용량 제한 (5MB)
  limits: {
    fileSize: 5 * 1024 * 1024,
  },
});

// localMiddleware 함수 작성
const localMiddleware = (req, res, next) => {
  res.locals.user = req.user || null;
  next();
};

// 로그인 상태를 res.locals에 저장하여 템플릿에서 사용 가능하도록 함
app.use(localMiddleware);

// ...

// 나머지 라우트 및 서버 설정 코드는 여기에 추가

// ...

app.listen(process.env.PORT, () => {
  console.log("listening on 8080");
});

app.use((req, res, next) => {
  // 로그인 상태를 res.locals에 저장하여 템플릿에서 사용 가능하도록 함
  res.locals.user = req.isAuthenticated();
  console.log(res.locals.user);
  next();
});

app.get("/", (req, res) => {
  console.log(req.user);
  res.render("index.ejs");
});

app.get("/register", (req, res) => {
  res.render("register.ejs");
});

app.post("/register", async (req, res) => {
  const { name, email, password, confirm_password } = req.body;

  if (password !== confirm_password) {
    // 비밀번호 확인이 일치하지 않으면 회원가입 중단
    return res.render("register", { error: "비밀번호 확인이 일치하지 않습니다." });
  }

  try {
    const existingUser = await User.findOne({ email });

    if (existingUser) {
      // 이미 등록된 이메일이면 회원가입 중단
      return res.render("register", { error: "이미 사용 중인 이메일입니다." });
    }

    const user = new User({ name, email });
    await User.register(user, password);
    res.redirect("/login");
  } catch (error) {
    console.log(error);
    res.redirect("/");
  }
});
app.post("/checkEmailDuplication", async (req, res) => {
  const { email } = req.body;

  // 이메일 중복 체크
  const existingUser = await User.findOne({ email });

  // 중복 여부에 따라 JSON 응답 보냄
  res.json({ isDuplicate: existingUser !== null });
});
app.get("/login", (req, res) => {
  res.render("login.ejs");
});

// 로그인을 처리하는 라우트에 passport.authenticate 미들웨어 사용
app.post(
  "/login",
  passport.authenticate("local", {
    failureRedirect: "/login", // 인증 실패 시 다시 /login으로 리디렉션
    successRedirect: "/", // 인증 성공 시 /로 리디렉션
  }),
  (req, res) => {
    // 이 부분은 인증이 성공하면 실행되는 부분입니다. (실행되지 않을 가능성이 높습니다.)
    // 만약 인증에 성공하면 successRedirect로 리디렉션되므로 이 부분은 실행되지 않을 것입니다.
    // 따라서 로그인 성공 시 어떤 동작을 취하고 싶다면 successRedirect로 리디렉션된 경로에서 처리해야 합니다.
  }
);

app.get("/logout", (req, res) => {
  req.logout(function (err) {
    if (err) {
      console.log(err);
    }
    // 로그아웃 성공 시 알림 창을 띄우고 메인 페이지로 리디렉션
    res.send('<script>alert("로그아웃 되었습니다."); window.location.href = "/";</script>');
  });
});

app.get("/upload", (req, res) => {
  res.render("upload.ejs");
});

app.post("/upload", uploadImage.array("file_object"), async (req, res, next) => {
  const fileUrls = req.files.map((file) => file.location); // 업로드된 파일들의 경로 정보는 req.files에 저장됨
  console.log(fileUrls); // 업로드된 파일들의 S3 경로를 출력

  // Flask API의 URL
  const FLASK_API_URL = "http://101.101.219.62:30006/process_image"; // 적절한 Flask API의 URL로 변경

  // POST 요청 보내기
  axios
    .post(FLASK_API_URL, {
      fileUrls: fileUrls,
      userEmail: req.user.email,
    })
    .then(function (response) {
      console.log("Flask 애플리케이션으로부터 받은 응답:", response.data);
      res.status(200).json({ message: "업로드가 완료되었습니다.", redirectUrl: "/album" });
    })
    .catch(function (error) {
      console.error("오류 발생:", error.message);
      res.status(500).json({ error: "Failed to save user." });
    });
});

// album 페이지 접근 시 로그인 상태 확인 후 처리
app.get("/album", async (req, res) => {
  if (!req.isAuthenticated()) {
    return res.send('<script>alert("로그인이 필요합니다."); window.location.href = "/login";</script>');
  }

  try {
    const user = await User.findOne({ email: req.user.email });
    if (!user) {
      return res.status(404).json({ error: "사용자를 찾을 수 없습니다." });
    }

    // 사용자 모델에 'imagePaths'라는 필드가 이미지 URL 또는 경로를 저장하고 있다고 가정합니다.
    const images = user.imagePaths; // 실제 모델에서 사용하는 필드명으로 'imagePaths' 대신 변경해주세요.

    res.render("album.ejs", { images: images });
  } catch (error) {
    console.error("이미지 가져오기 에러:", error);
    res.status(500).json({ error: "이미지를 가져오는데 실패했습니다." });
  }
});

// 추억찾기 페이지 접근 시 로그인 상태 확인 후 처리
app.get("/retrieval", (req, res) => {
  if (req.isAuthenticated()) {
    // 로그인 상태라면 추억찾기 페이지로 이동
    res.render("retrieval.ejs");
  } else {
    // 로그인되어 있지 않은 상태라면 알림창을 띄우고 로그인 페이지로 리디렉션
    res.send('<script>alert("로그인이 필요합니다."); window.location.href = "/login";</script>');
  }
});

// 추억찾기 페이지 접근 시 로그인 상태 확인 후 처리
app.get("/home", (req, res) => {
  res.render("home.ejs");
});

app.get("/profile", (req, res) => {
  // 현재 사용자 정보를 가져와서 프로필 페이지에 렌더링
  const currentUser = req.user;
  res.render("profile", { user: currentUser });
});

async function uploadImagePathsToFlaskAPI(email, imagePaths) {
  try {
    // Flask API의 엔드포인트 URL
    const apiUrl = "http://101.101.219.62:30006/inference";

    // 요청할 데이터 생성
    const requestData = {
      email: email,
      imagePaths: imagePaths,
    };

    // Flask API에 POST 요청 보내기
    const response = await axios.post(apiUrl, requestData);

    // Flask API의 응답 처리
    if (response.status === 200) {
      console.log("이미지 경로들을 Flask API에 전송하였습니다.");
      console.log(response.data); // Flask API의 응답 데이터 확인
      return response.data;
    } else {
      console.error("Flask API 요청이 실패하였습니다.");
      return null;
    }
  } catch (error) {
    console.error("Flask API 요청 중 오류 발생:", error.message);
    return null;
  }
}

//get("/retrieval") 라우트 핸들러 예시
app.get("/retrieval", async (req, res) => {
  try {
    // req.user에서 email 가져오기 (이 부분은 실제로는 로그인을 통해 인증된 사용자 정보에서 가져오는 로직으로 대체해야 합니다.)
    const userEmail = req.user.email; // 사용자의 이메일 정보를 가져와서 저장합니다.

    try {
      const user = await User.findOne({ email: req.user.email });
      if (!user) {
        return res.status(404).json({ error: "사용자를 찾을 수 없습니다." });
      }

      // 사용자 모델에 'imagePaths'라는 필드가 이미지 URL 또는 경로를 저장하고 있다고 가정합니다.
      const requestData = user.imagePaths; // 실제 모델에서 사용하는 필드명으로 'imagePaths' 대신 변경해주세요.

      // Flask API에 POST 요청 보내기
      const apiResponse = await uploadImagePathsToFlaskAPI(userEmail, requestData);

      // API 응답에 따라 처리
      if (apiResponse && apiResponse.message === "POST 요청이 성공적으로 처리되었습니다.") {
        // API 요청이 성공적으로 처리된 경우
        console.log("Flask API에서의 처리가 성공하였습니다.");
        // 다음 로직 추가: Flask API에서 반환한 결과를 이용하여 추가 처리
        res.status(200).json({ message: "Flask API 처리 성공" });
      } else {
        // API 요청이 실패한 경우 또는 API 응답에 에러 정보가 있는 경우
        console.error("Flask API에서 처리 실패 또는 오류 발생");
        res.status(500).json({ error: "Flask API 처리 실패" });
      }
    } catch (error) {
      console.error("이미지 가져오기 에러:", error);
      res.status(500).json({ error: "이미지를 가져오는데 실패했습니다." });
    }
  } catch (error) {
    console.error("에러 발생:", error.message);
    res.status(500).json({ error: "서버 에러 발생" });
  }
});
