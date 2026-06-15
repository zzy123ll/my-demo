
import hashlib, secrets, os as _os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), unique=True, nullable=False, index=True)
    username = Column(String(128), nullable=False)
    password_hash = Column(String(256), nullable=False)
    salt = Column(String(64), nullable=False)
    role = Column(String(16), default="staff")
    department = Column(String(64), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, index=True)
    user_id = Column(String(64), nullable=False)
    status = Column(String(16), default="active")
    summary = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    user_id = Column(String(64))
    role = Column(String(16))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(64), unique=True, index=True)
    session_id = Column(String(64))
    user_id = Column(String(64))
    user_name = Column(String(128))
    query = Column(Text)
    status = Column(String(16), default="pending")
    assigned_agent = Column(String(64), default="")
    resolution = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

def hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return salt + ":" + h.hex(), salt

def verify_password(password: str, stored: str) -> bool:
    salt = stored.split(":")[0]
    new_hash, _ = hash_password(password, salt)
    return new_hash == stored

def init_db(path="sqlite:///data/rag_cs.db"):
    _os.makedirs("data", exist_ok=True)
    engine = create_engine(path, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    if not db.query(User).filter_by(user_id="admin").first():
        pw, salt = hash_password("admin123")
        db.add(User(user_id="admin", username="Administrator", password_hash=pw, salt=salt, role="admin"))
        pw2, s2 = hash_password("agent123")
        db.add(User(user_id="agent01", username="Agent Wang", password_hash=pw2, salt=s2, role="agent"))
        pw3, s3 = hash_password("staff123")
        db.add(User(user_id="zhangsan", username="Zhang San", password_hash=pw3, salt=s3, role="staff", department="Engineering"))
        db.commit()
    db.close()
    return engine, Session
