from fastapi import APIRouter

router = APIRouter()


@router.get("/_shim/galaxy")
def _galaxy_shim():
    return {"msg": "galaxy shim"}
