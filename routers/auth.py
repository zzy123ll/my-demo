
import hashlib, secrets, time, json, base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database.models import User, hash_password, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Simple JWT without external dependency
def make_token(user_id, role):
    header = base64.urlsafe_b64encode(json.dumps({"alg":"HS256"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"user_id":user_id,"role":role,"exp":int(time.time())+86400}).encode()).decode().rstrip("=")
    sig = hashlib.sha256(f"{header}.{payload}.secret-rag-key".encode()).hexdigest()[:32]
    return f"{header}.{payload}.{sig}"

def decode_token(token):
    try:
        parts = token.split(".")
        if len(parts) != 3: return None
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
        sig = hashlib.sha256(f"{parts[0]}.{parts[1]}.secret-rag-key".encode()).hexdigest()[:32]
        if sig != parts[2]: return None
        if payload.get("exp", 0) < time.time(): return None
        return payload
    except: return None

class LoginReq(BaseModel):
    user_id: str
    password: str

class RegisterReq(BaseModel):
    user_id: str
    username: str
    password: str
    department: str = ""

@router.post("/login")
async def login(req: LoginReq):
    from database.models import init_db
    _, Session = init_db()
    db = Session()
    user = db.query(User).filter_by(user_id=req.user_id).first()
    db.close()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = make_token(user.user_id, user.role)
    return {"token": token, "user": {"user_id": user.user_id, "username": user.username, "role": user.role, "department": user.department}}

@router.post("/register")
async def register(req: RegisterReq):
    _, Session = init_db()
    db = Session()
    if db.query(User).filter_by(user_id=req.user_id).first():
        db.close()
        raise HTTPException(400, "User already exists")
    pw, salt = hash_password(req.password)
    user = User(user_id=req.user_id, username=req.username, password_hash=pw, salt=salt, role="staff", department=req.department)
    db.add(user); db.commit(); db.close()
    token = make_token(req.user_id, "staff")
    return {"token": token, "user": {"user_id": req.user_id, "username": req.username, "role": "staff"}}

def get_current_user(request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    payload = decode_token(auth[7:])
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    return payload
