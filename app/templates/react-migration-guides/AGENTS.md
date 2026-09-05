# Shopping AI Assistant React 마이그레이션 지침

## 역할

이 프로젝트에서는 숙련된 React 프론트엔드 개발자로 작업한다. 현재의 단일 HTML/CSS/JavaScript 구현을 유지보수 가능한 React 애플리케이션으로 마이그레이션하는 것이 최우선 목표다.

## 작업 전 필수 확인

1. 코드 수정 전에 `REACT_MIGRATION_SPEC.md`를 읽는다.
2. `MIGRATION_PLAN.md`의 단계와 범위를 따른다.
3. 작업 완료 전에 `VERIFICATION_CHECKLIST.md`를 전부 확인한다.
4. 원본 HTML과 기존 프로젝트 파일을 먼저 조사하고, 확인하지 않은 프레임워크·빌드 명령·API 계약을 추측하지 않는다.
5. 기존 작업물이 있으면 덮어쓰지 말고 현재 구조와 규칙을 우선한다.

## 핵심 원칙

- 마이그레이션 중 화면과 사용자 동작을 임의로 재설계하지 않는다.
- React 함수 컴포넌트와 Hooks를 사용한다.
- 직접적인 DOM 생성·삭제·수정 대신 React state와 props로 렌더링한다.
- `innerHTML`, 인라인 `onclick`, `window.__removeFile` 같은 전역 핸들러를 사용하지 않는다.
- 사용자 입력과 API 응답을 JSX 텍스트로 렌더링해 기본 이스케이프를 유지한다.
- 상태는 명확히 구분한다: 입력값, 선택 파일, 미리보기 URL, 세션 ID, 메시지 목록, 요청 진행 상태, 오류.
- 컴포넌트는 역할별로 분리하되, 단순한 UI까지 과도하게 추상화하지 않는다.
- Blob URL은 더 이상 필요하지 않을 때 `URL.revokeObjectURL`로 해제한다.
- 요청 진행 중 중복 전송을 막는다.
- 이미지가 없어도 텍스트 검색이 가능하고, 텍스트가 없어도 이미지 검색이 가능해야 한다.
- API URL이나 응답 필드 이름을 근거 없이 변경하지 않는다.
- 새 의존성은 꼭 필요한 경우에만 추가하고 이유를 설명한다.

## 권장 컴포넌트 구조

프로젝트의 기존 구조가 없다면 다음을 기본안으로 사용한다.

```text
src/
  api/
    shoppingApi.js
  components/
    ChatLog.jsx
    ChatMessage.jsx
    Composer.jsx
    ImagePreview.jsx
    ProductCard.jsx
    ProductGrid.jsx
  hooks/
    useObjectUrl.js
  App.jsx
  main.jsx
  styles.css
```

필요하면 구조를 조정할 수 있지만, API 호출과 화면 렌더링 책임은 분리한다.

## 구현 규칙

- 리스트 렌더링에는 안정적인 `key`를 사용한다. 가능하면 상품 ID나 링크를 사용하고 배열 인덱스만으로 식별하지 않는다.
- `data.results`가 없거나 배열이 아니어도 화면이 깨지지 않도록 정규화한다.
- API 오류는 사용자에게 한국어 메시지로 표시하고 개발자용 세부 내용은 콘솔 또는 별도 로깅으로 남긴다.
- 외부 링크는 유효한 URL일 때만 열고 `noopener,noreferrer`를 적용한다.
- 이미지에는 의미 있는 `alt`를 제공하고 로드 실패 상태를 처리한다.
- 첨부 버튼과 삭제 버튼에는 접근 가능한 이름을 제공한다.
- Enter는 전송, Shift+Enter는 줄바꿈 동작을 유지한다. IME 조합 중 Enter로 전송되지 않게 한다.
- CSS는 기존 시각적 결과를 우선 보존하고, ID 선택자는 재사용 가능한 class 중심으로 전환한다.
- 환경별 API 주소가 필요하면 Vite 환경 변수 사용을 고려하되, 기존 동작을 먼저 보존한다.

## 테스트와 검증

- 가능한 경우 파일 선택·삭제, 텍스트 전송, Enter/Shift+Enter, 로딩, 성공, 빈 결과, API 실패를 테스트한다.
- 기존 프로젝트의 lint, test, build 명령을 사용한다.
- 명령이 없다면 `package.json`을 확인한 후 적절한 최소 검증 방법을 제안하거나 구성한다.
- 실제 API 호출이 어려우면 fetch를 mock하여 UI 상태 전이를 검증한다.

## 완료 보고 형식

작업 완료 시 다음 내용을 간단히 보고한다.

1. 변경된 구조와 핵심 구현
2. 원본에서 보존한 동작
3. 실행한 검증 명령과 결과
4. 확인하지 못한 항목 또는 남은 위험

