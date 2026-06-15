
const API_BASE = "";
let AUTH_TOKEN = localStorage.getItem("rag_token") || "";

function setToken(t) { AUTH_TOKEN = t; localStorage.setItem("rag_token", t); }
function getToken() { return AUTH_TOKEN; }
function clearToken() { AUTH_TOKEN = ""; localStorage.removeItem("rag_token"); }

async function api(method, path, body = null) {
    const headers = { "Content-Type": "application/json" };
    if (AUTH_TOKEN) headers["Authorization"] = "Bearer " + AUTH_TOKEN;
    const opts = { method, headers };
    if (body) opts.body = JSON.stringify(body);
    const r = await fetch(API_BASE + path, opts);
    if (r.status === 401) { clearToken(); window.location.href = "/static/login.html"; throw new Error("Unauthorized"); }
    if (!r.ok) { const e = await r.json().catch(()=>({})); throw new Error(e.detail || "Request failed"); }
    return r.json();
}

function escapeHtml(s) {
    if (!s) return "";
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
