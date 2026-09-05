# React 마이그레이션 명세

## 목표

단일 HTML 파일로 구현된 `Shopping AI Assistant`를 React 애플리케이션으로 옮긴다. 마이그레이션의 성공 기준은 기존 사용자 기능과 API 흐름을 보존하면서 수동 DOM 조작을 React의 선언적 상태 관리로 대체하는 것이다.

## 현재 기능

### 채팅 UI

- 사용자 메시지는 오른쪽 말풍선으로 표시한다.
- 어시스턴트 메시지는 왼쪽 말풍선으로 표시한다.
- 메시지가 추가되면 채팅 로그를 아래로 스크롤한다.
- 어시스턴트 메시지에는 상태 문구와 상품 결과 그리드가 들어간다.

### 입력 및 이미지 첨부

- 텍스트 입력과 이미지 파일 첨부를 지원한다.
- 첨부 버튼을 누르면 숨겨진 이미지 파일 입력을 연다.
- 선택한 이미지를 64×64 미리보기로 표시한다.
- 미리보기의 삭제 버튼으로 선택 파일을 제거한다.
- Enter는 전송하고 Shift+Enter는 줄바꿈한다.
- 텍스트와 이미지가 모두 없으면 전송하지 않는다.

### 이미지 업로드

1. 이미지가 있으면 `POST {API_BASE}/products`로 presigned URL을 요청한다.
2. 요청 JSON에는 `content_type`과 현재 `session_id`를 보낸다.
3. 응답에 새 `session_id`가 있으면 현재 세션 상태에 저장한다.
4. 받은 `presigned_url`에 이미지 파일을 `PUT`한다.
5. 업로드가 끝나면 응답의 `s3_key`를 상품 검색 요청에 전달한다.

현재 원본의 API 기준값:

```text
API_BASE=https://70crv2wl9a.execute-api.ap-northeast-2.amazonaws.com/dev
```

### 상품 검색

- `POST /search?top_k=8`에 `FormData`를 보낸다.
- 폼 필드:
  - `session_id`
  - `message` — 값이 있을 때
  - `s3_key` — 값이 있을 때
- 성공 응답에서 `answer`와 `results`를 사용한다.
- `results`가 비었고 `answer`도 없으면 `유사한 상품을 찾지 못했습니다.`를 표시한다.
- 검색 중에는 전송 버튼을 비활성화한다.
- 실패하면 `요청 실패: {오류 메시지}`를 표시한다.

> 주의: 원본은 presigned 요청에는 원격 `API_BASE`를 사용하지만 검색 요청에는 상대 경로 `/search`를 사용한다. 이는 의도된 프록시 구성일 수 있으므로, 백엔드 또는 개발 서버 구성을 확인하기 전에는 임의로 하나의 주소로 통합하지 않는다.

### 상품 결과 카드

각 상품은 다음 정보를 표시한다.

- `image_url`
- `title`
- `hprice` 또는 `??`
- `lprice` 또는 `??`
- `mall_name`
- 클릭 가능한 `link`

기존 초기 렌더링 코드에 존재하는 `product_id`, `score`도 API가 실제로 반환하는지 확인한다. 최종 UI 필드는 실제 응답 계약을 기준으로 통일한다.

## React 상태 모델

최소한 다음 상태를 명시적으로 관리한다.

```text
messageInput: string
selectedFile: File | null
previewUrl: string | null
sessionId: string | null
messages: ChatMessage[]
isSubmitting: boolean
```

권장 메시지 모델:

```js
{
  id: string,
  role: 'user' | 'assistant',
  text: string,
  imageUrl?: string,
  status?: 'loading' | 'success' | 'error',
  results?: Product[]
}
```

로딩 어시스턴트 메시지는 먼저 추가한 뒤 동일한 `id`를 이용해 상태와 결과를 갱신한다.

## 비기능 요구사항

- 기존 최대 너비 900px와 전체적인 채팅 레이아웃을 유지한다.
- 모바일에서도 composer와 상품 그리드가 화면 밖으로 넘치지 않아야 한다.
- 키보드만으로 첨부, 입력, 전송, 파일 제거, 상품 링크 접근이 가능해야 한다.
- 사용자 입력 또는 API 문자열을 HTML로 주입하지 않는다.
- 컴포넌트 unmount와 파일 교체 시 Blob URL을 정리한다.
- 네트워크 요청의 성공 여부를 `response.ok`로 확인한다.
- 예상하지 못한 응답 형태를 방어적으로 처리한다.

## 이번 범위에서 제외

- UI 전면 재디자인
- 백엔드 API 변경
- 로그인 또는 영구 세션 저장 도입
- 상태관리 라이브러리 추가
- 서버 렌더링 프레임워크로의 전환

해당 항목은 사용자가 별도로 요청할 때만 진행한다.

