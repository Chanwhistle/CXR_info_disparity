#!/bin/bash

# Google Drive 업로드 스크립트
# 사용법: ./upload_to_gdrive.sh

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Google Drive 업로드 스크립트 ===${NC}"
echo ""

# rclone이 설치되어 있는지 확인
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}오류: rclone이 설치되어 있지 않습니다.${NC}"
    echo "설치 방법: sudo apt-get install rclone 또는 https://rclone.org/install/"
    exit 1
fi

# rclone 설정 확인
if ! rclone listremotes &> /dev/null; then
    echo -e "${YELLOW}rclone이 설정되어 있지 않습니다.${NC}"
    echo "설정을 시작합니다..."
    echo ""
    echo "다음 명령어를 실행하여 구글 드라이브를 설정하세요:"
    echo "  rclone config"
    echo ""
    echo "설정 시:"
    echo "  1. 'n' (새 remote 생성)"
    echo "  2. 이름 입력 (예: gdrive)"
    echo "  3. storage provider 선택: 'drive' 선택 (번호 입력)"
    echo "  4. client_id, client_secret: Enter로 건너뛰기 가능"
    echo "  5. scope: 'drive' 선택"
    echo "  6. root_folder_id: Enter로 건너뛰기"
    echo "  7. service_account_file: Enter로 건너뛰기"
    echo "  8. auth process를 위해 웹 브라우저에서 인증"
    echo ""
    exit 1
fi

# Remote 이름 입력 받기
echo -e "${YELLOW}사용 가능한 remotes:${NC}"
AVAILABLE_REMOTES=$(rclone listremotes | head -1 | cut -d: -f1)
rclone listremotes

echo ""
# 기본값으로 첫 번째 remote 사용 (또는 명령행 인자)
if [ -n "$1" ] && [ "$1" != "--dry-run" ]; then
    REMOTE_NAME="$1"
elif [ -n "$AVAILABLE_REMOTES" ]; then
    REMOTE_NAME="$AVAILABLE_REMOTES"
    echo -e "${YELLOW}기본값 사용: $REMOTE_NAME${NC}"
    read -p "다른 remote를 사용하려면 이름을 입력하세요 (Enter로 기본값 사용): " USER_INPUT
    if [ -n "$USER_INPUT" ]; then
        REMOTE_NAME="$USER_INPUT"
    fi
else
    read -p "사용할 remote 이름을 입력하세요 (예: gdrive): " REMOTE_NAME
    if [ -z "$REMOTE_NAME" ]; then
        REMOTE_NAME="gdrive"
        echo -e "${YELLOW}기본값 사용: $REMOTE_NAME${NC}"
    fi
fi

# Google Drive에 테스트 연결
echo ""
echo -e "${GREEN}구글 드라이브 연결 테스트 중...${NC}"
if ! rclone lsd "$REMOTE_NAME:" &> /dev/null; then
    echo -e "${RED}오류: $REMOTE_NAME에 연결할 수 없습니다.${NC}"
    echo "rclone config를 다시 실행하여 설정을 확인하세요."
    exit 1
fi

echo -e "${GREEN}연결 성공!${NC}"
echo ""

# 업로드할 폴더 경로
LOCAL_PATH="."
REMOTE_PATH="${REMOTE_NAME}:CXR_info_disparity"

echo -e "${YELLOW}업로드 정보:${NC}"
echo "  로컬 경로: $(pwd)"
echo "  원격 경로: $REMOTE_PATH"
echo ""

# .gitignore 기반 필터 파일 생성
FILTER_FILE="/tmp/rclone-filters.txt"
cat > "$FILTER_FILE" << 'EOF'
# .gitignore 기반 필터
- __pycache__/
- *.pyc
- *.pyo
- *.pyd
- .Python
- env/
- venv/
- ENV/
- *.egg-info/
- .pytest_cache/
- .ipynb_checkpoints/
- .DS_Store
- *.log
- .vscode/
- .idea/
- *.swp
- *.swo
- *~
- wandb/
- analyze/
- inference_result/
- dataset/
- cache_images/
- physionet.org/
- saved_images/
- attention_outputs/
- trained_models/
- *.bin
- *.pt
- *.pth
- former/
- __pycache__
EOF

# .dockerignore의 내용도 추가 (데이터 파일 제외)
echo "" >> "$FILTER_FILE"
echo "# .dockerignore 기반 필터" >> "$FILTER_FILE"
echo "- dataset/" >> "$FILTER_FILE"
echo "- physionet.org/" >> "$FILTER_FILE"
echo "- saved_images/" >> "$FILTER_FILE"
echo "- trained_models/" >> "$FILTER_FILE"
echo "- cache_images/" >> "$FILTER_FILE"
echo "- attention_outputs/" >> "$FILTER_FILE"
echo "- inference_result/" >> "$FILTER_FILE"

echo -e "${YELLOW}필터 규칙이 적용됩니다 (대용량 데이터 폴더 제외).${NC}"
echo ""

# 확인
read -p "업로드를 시작하시겠습니까? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 0
fi

echo ""
echo -e "${GREEN}업로드 시작...${NC}"
echo "이 작업은 시간이 걸릴 수 있습니다."
echo ""

# rclone copy 사용 (증분 업로드, --dry-run으로 먼저 테스트 가능)
# 실제 업로드 전에 --dry-run으로 테스트할 수 있습니다
DRY_RUN_FLAG=false
if [ "$1" == "--dry-run" ] || [ "$2" == "--dry-run" ]; then
    DRY_RUN_FLAG=true
fi

if [ "$DRY_RUN_FLAG" == "true" ]; then
    echo -e "${YELLOW}[DRY RUN] 실제로 업로드되지 않습니다.${NC}"
    rclone copy "$LOCAL_PATH" "$REMOTE_PATH" \
        --filter-from "$FILTER_FILE" \
        --dry-run \
        --progress \
        --verbose
else
    rclone copy "$LOCAL_PATH" "$REMOTE_PATH" \
        --filter-from "$FILTER_FILE" \
        --progress \
        --verbose \
        --transfers 4 \
        --checkers 8 \
        --drive-chunk-size 128M
    
    echo ""
    echo -e "${GREEN}업로드 완료!${NC}"
    echo ""
    echo "업로드된 파일 확인:"
    echo "  rclone ls $REMOTE_PATH"
    echo ""
    echo "동기화 상태 확인:"
    echo "  rclone check $LOCAL_PATH $REMOTE_PATH"
fi

# 임시 파일 삭제
rm -f "$FILTER_FILE"
