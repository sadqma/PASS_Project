//
// ===================== GLOBAL STATE =====================
//
let dashboardData = null;
let chart = null;
let currentFilter = "all";   // текущий фильтр (all/safe/medium/risk)
window.currentStudent = null; // текущий студент


//
// ===================== DOM ELEMENTS =====================
//
const loginForm = document.getElementById("login-form");
const loginSection = document.getElementById("login-section");
const dashboard = document.getElementById("dashboard");
const summaryCard = document.getElementById("summary-card");
const errorDiv = document.getElementById("error");

const studentLabel = document.getElementById("dash-student");
const coursesTable = document.getElementById("courses-table");
const adviceList = document.getElementById("advice-list");

const avgSuccessEl = document.getElementById("avg-success");
const courseCountEl = document.getElementById("course-count");
const riskCountEl = document.getElementById("risk-count");


//
// ===================== LOAD JSON =====================
//
async function loadDashboardData() {
  try {
    const resp = await fetch("dashboard_data.json");
    if (!resp.ok) throw new Error("dashboard_data.json missing");

    dashboardData = await resp.json();
    console.log("Dashboard data loaded");
  } catch (err) {
    errorDiv.textContent =
      "Не удалось загрузить dashboard_data.json. Запусти сервер: python3 -m http.server";
    errorDiv.classList.remove("hidden");
    console.error(err);
  }
}

loadDashboardData();


//
// ===================== LOGIN HANDLER =====================
//
loginForm.addEventListener("submit", (e) => {
  e.preventDefault();

  errorDiv.classList.add("hidden");
  dashboard.classList.add("hidden");
  summaryCard.classList.add("hidden");

  if (!dashboardData) {
    errorDiv.textContent = "Данные ещё загружаются...";
    errorDiv.classList.remove("hidden");
    return;
  }

  const studentId = document.getElementById("student_id").value.trim();
  if (!studentId) return;

  const student = dashboardData.students[studentId];
  if (!student) {
    errorDiv.textContent = "Данные для этого Student ID не найдены.";
    errorDiv.classList.remove("hidden");
    return;
  }

  window.currentStudent = student;
  renderDashboard(student);

  // анимации
  loginSection.classList.add("fade-out");
  setTimeout(() => loginSection.classList.add("hidden"), 350);

  summaryCard.classList.remove("hidden");
  summaryCard.classList.add("fade-in");
  setTimeout(() => summaryCard.classList.add("visible"), 10);

  dashboard.classList.remove("hidden");
  dashboard.classList.add("fade-in");
  setTimeout(() => dashboard.classList.add("visible"), 10);
});


//
// ===================== RENDER DASHBOARD =====================
//
function renderDashboard(student) {
  let courses = student.courses;

  // === APPLY FILTER ===
  courses = courses.filter((c) => {
    if (currentFilter === "all") return true;
    if (currentFilter === "safe") return c.prob_success >= 0.8;
    if (currentFilter === "medium")
      return c.prob_success >= 0.6 && c.prob_success < 0.8;
    if (currentFilter === "risk") return c.prob_success < 0.6;
  });

  // Если курсов нет
  if (courses.length === 0) {
    coursesTable.innerHTML = "";
    adviceList.innerHTML = "";
    renderChart([]);
    avgSuccessEl.textContent = "0%";
    courseCountEl.textContent = "0";
    riskCountEl.textContent = "0";
    return;
  }

  studentLabel.textContent = `Student ID: ${student.student_id}`;

  // Очищаем таблицу и рекомендации
  coursesTable.innerHTML = "";
  adviceList.innerHTML = "";

  let adviceSet = new Set();
  let sumProb = 0;
  let riskCount = 0;

  courses.forEach((course) => {
    const tr = document.createElement("tr");

    const final = course.final_calc;
    const prob = course.prob_success;
    const letter = final != null ? percentToLetter(final) : "–";

    const badgeClass =
      prob >= 0.8 ? "badge-good" :
      prob >= 0.6 ? "badge-mid" : "badge-bad";

    tr.innerHTML = `
      <td>${course.discipline}</td>
      <td>${final != null ? final.toFixed(1) + "%" : "—"}</td>
      <td><span class="badge ${badgeClass}">${course.prob_success_percent}%</span></td>
      <td>${letter}</td>
    `;

    coursesTable.appendChild(tr);

    sumProb += prob;
    if (prob < 0.6) riskCount++;

    if (course.advice) adviceSet.add(course.advice);
  });

  // Summary
  const avg = (sumProb / courses.length) * 100;
  avgSuccessEl.textContent = avg.toFixed(1) + "%";
  courseCountEl.textContent = courses.length;
  riskCountEl.textContent = riskCount;

  // Recommendations
  adviceList.innerHTML = "";
  adviceSet.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    adviceList.appendChild(li);
  });

  // Chart
  renderChart(courses);
}



//
// ===================== CHART (FIXED VERSION) =====================
//
function renderChart(courses) {
  const ctx = document.getElementById("successChart").getContext("2d");

  const labels = courses.map((c) => c.discipline);
  const values = courses.map((c) => c.prob_success_percent);

  if (chart) chart.destroy();

  // === SAFE Y LIMITS (0..100) + MARGIN ===
  let minVal = Math.min(...values, 0);
  let maxVal = Math.max(...values, 100);

  minVal = Math.max(0, minVal - 5);
  maxVal = Math.min(100, maxVal + 5);

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Success probability (%)",
          data: values,
          borderColor: "#7EB0FF",
          backgroundColor: "rgba(126,176,255,0.15)",
          pointBackgroundColor: "#7EB0FF",
          pointRadius: 5,
          pointHoverRadius: 7,
          borderWidth: 2,
          tension: 0.35,
        },
      ],
    },

    options: {
      responsive: true,
      maintainAspectRatio: false,

      scales: {
        y: {
          min: minVal,
          max: maxVal,
          grid: { color: "#E2E8F0" },
          ticks: { color: "#64748B" }
        },
        x: {
          offset: true,   // центрирует точку если она одна
          grid: { display: false },
          ticks: { color: "#64748B" }
        }
      },

      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#FFFFFF",
          borderColor: "#CBD5E1",
          borderWidth: 1,
          titleColor: "#1E293B",
          bodyColor: "#1E293B",
          padding: 10,
          displayColors: false,
        },
      },
    },
  });
}



//
// ===================== FILTER BUTTONS =====================
//
document.addEventListener("click", (e) => {
  if (!e.target.classList.contains("filter-btn")) return;

  document.querySelectorAll(".filter-btn").forEach((btn) =>
    btn.classList.remove("active")
  );

  e.target.classList.add("active");
  currentFilter = e.target.dataset.filter;

  if (window.currentStudent) {
    renderDashboard(window.currentStudent);
  }
});



//
// ===================== HELPERS =====================
//
function percentToLetter(pct) {
  if (pct >= 90) return "A";
  if (pct >= 80) return "B";
  if (pct >= 70) return "C";
  if (pct >= 60) return "D";
  return "F";
}
