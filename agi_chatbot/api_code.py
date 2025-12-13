from fastapi import APIRouter

# Minimal shim for api_code router used during in-process testing.
router = APIRouter()


@router.get("/_shim/api_code/hello")
def _hello():
    return {"msg": "api_code shim"}
